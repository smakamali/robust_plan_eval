####################### TEST ROQ #######################

# TODO: change best model path files to text
# TODO: read model configs from disk

import os
import pickle
import torch
import numpy as np
import pandas as pd
from pyg_data import queryPlanPGDataset
from util.util import set_seed
from util.data_transform import *
from util.custom_loss import aleatoric_loss, rmse_loss
import pytorch_lightning as pl
from lcm.roq_model import lcm_pl as roq
from lcm.neo_bao_model import lcm_pl as neo_bao
from torch_geometric.loader import DataLoader
from util.torch_util import LitMCdropoutModel
from util.eval_util import *


num_workers = 10

models_path = os.path.join('.','lightning_models')

roq_config ={'TCNNin': 64,
 'TCNNout': 32,
 'batch_size': 512,
 'dropout': 0.12333583157720673,
 'finalMLPin': 128,
 'finalMLPout': 128,
 'lr': 0.00035,
 'node_embd_dim': 16,
 'qp_attheads': 1,
 'qp_gnn_ls': 3,
 'query_module_out': 64}

neo_config={'TCNNin': 512,
 'TCNNout': 128,
 'batch_size': 512,
 'dropout': 0.08178909250467073,
 'finalMLPin': 128,
 'finalMLPout': 32,
 'lr': 0.001,
 'node_embd_dim': 16,
 'qp_attheads': 1,
 'qp_gnn_ls': 3,
 'query_module_out': 32,
 'query_module_in': 128}

bao_config={'TCNNin': 256,
 'TCNNout': 64,
 'batch_size': 512,
 'dropout': 0.15,
 'finalMLPin': 64,
 'finalMLPout': 32,
 'lr': 0.0005,
 'node_embd_dim': 16,
 'qp_attheads': 1,
 'qp_gnn_ls': 3,
 'query_module_out': 64}

def load_best_model_paths(architecture_p,experiment_id):
    with open(
        os.path.join(models_path,'best_model_paths_{}_{}.pkl'.format(architecture_p,experiment_id)), 'rb') as file:
        return pickle.load(file)
    
def test(experiment_id = 'job',architecture_p = 'roq',
         files_id = 'temp', try_gpu_inf=True, seed = 0, num_workers=num_workers, show_fig = False):

    set_seed(seed)

    # setting torch.device()
    if try_gpu_inf:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    if try_gpu_inf and str(device) != 'cuda':
        print("Warning: GPU was not available for inferenec. Using CPU instead.")
    print("Device used for inference:",str(device).upper())

    # create results dir if it does not exist
    if not os.path.exists('./results'):
        os.mkdir('./results')

    # loading model paths
    best_model_paths = load_best_model_paths('roq',experiment_id)
    neo_paths =  load_best_model_paths('neo',experiment_id)
    bao_paths =  load_best_model_paths('bao',experiment_id)

    # Load train, validation, and test datasets
    print("loading train")
    train_set = queryPlanPGDataset(
        split= 'train', files_id = files_id,
        force_reload=False, seed = seed 
        )
    print("Number of queries in training dataset: ",train_set.len())
    
    print("loading val")
    val_set = queryPlanPGDataset(
        split= 'val', files_id = files_id
        )
    print("Number of queries in vlidation dataset: ",val_set.len())
    
    print("loading test")
    test_set = queryPlanPGDataset(
        split= 'test', files_id = files_id
        )
    print("Number of queries in test dataset: ",test_set.len())

    # Perform data transformations on inputs 
    drop_const = dropConst(train_set)
    train_set = drop_const(train_set)
    val_set = drop_const(val_set)
    test_set = drop_const(test_set)

    null_imp = nullImputation(train_set)
    train_set = null_imp(train_set)
    val_set = null_imp(val_set)
    test_set = null_imp(test_set)

    minmax_scale = minmaxScale(train_set)
    train_set = minmax_scale(train_set)
    val_set = minmax_scale(val_set)
    test_set = minmax_scale(test_set)

    # Perform data transformations on targets 
    if 'roq' in architecture_p:
        yTransFunc = target_log_transform(train_set)
    else:
        yTransFunc = target_transform(train_set)

    train_set = yTransFunc.transform(train_set)
    val_set = yTransFunc.transform(val_set)
    test_set = yTransFunc.transform(test_set)

    # Capture node, edge, graph, and plan attribute shapes to initialize the model 
    plan_attr_shape = train_set[0].plan_attr.shape
    plan_ord_shape = train_set[0].plan_ord.shape
    graph_attr_shape= train_set[0].graph_attr.shape
    edge_attr_shape = train_set[0].edge_attr_s.shape
    node_attr_shape = train_set[0].x_s.shape

    print("plan_attr_shape",plan_attr_shape)
    print("plan_ord_shape",plan_ord_shape)
    print("graph_attr_shape",graph_attr_shape)
    print("edge_attr_shape",edge_attr_shape)
    print("node_attr_shape",node_attr_shape)

    batch_size= roq_config['batch_size']

    follow_batch = ['x_s']

    # train_loader = DataLoader(
    #     train_set, batch_size=batch_size,
    #     shuffle=False, num_workers=0, 
    #     follow_batch=follow_batch,
    #     #  persistent_workers=True
    #     )
    val_loader = DataLoader(
        val_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, follow_batch=follow_batch,
        #  persistent_workers=True
        )
    test_loader = DataLoader(
        test_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, follow_batch=follow_batch,
        #  persistent_workers=True
        )
    
    aleatoric_l = aleatoric_loss(device=device)
    rmse_l = rmse_loss()

    torch.set_float32_matmul_precision('high')

    qerror_dict = {}
    corr_dict = {}
    rt_res_dict = {}
    so_res_dict = {}
    agg_res_dict = {}

    # For loop starts here

    for i,_ in enumerate(best_model_paths):
        best_model_path = best_model_paths[i]
        neo_path = neo_paths[i]
        bao_path = bao_paths[i]

        # Note: model paramteres must be passed unless they are the same of defaults
        model = roq.load_from_checkpoint(
            best_model_path, 
            num_node = node_attr_shape[0], 
            node_dim = node_attr_shape[1],
            edge_dim = edge_attr_shape[1],#fill_value =0, 
            numPlanFeat=plan_attr_shape,
            numPlanOrdFeat=plan_ord_shape,
            numQueryGraphFeat = graph_attr_shape[0],
            with_var = True, device = device, 
            criterion = aleatoric_l,
            **roq_config
            )

        neo_vanilla = neo_bao.load_from_checkpoint(
            neo_path,
            num_node = node_attr_shape[0], 
            node_dim = node_attr_shape[1],
            edge_dim = edge_attr_shape[1],
            numPlanFeat=plan_attr_shape,
            numPlanOrdFeat=plan_ord_shape,
            numQueryGraphFeat = graph_attr_shape[0],
            device = device, 
            criterion = rmse_l,
            architecture = 'neo',
            **neo_config
            )

        bao_vanilla = neo_bao.load_from_checkpoint(
            bao_path,
            num_node = node_attr_shape[0], 
            node_dim = node_attr_shape[1],
            edge_dim = edge_attr_shape[1],
            numPlanFeat=plan_attr_shape,
            numPlanOrdFeat=plan_ord_shape,
            numQueryGraphFeat = graph_attr_shape[0],
            device = device, 
            criterion = rmse_l,
            architecture = 'bao',
            **bao_config
            )

        # Enables variational inference using MC dropout
        LitMCDPModel = LitMCdropoutModel(
            model, mc_iteration=10, 
            dropout=roq_config['dropout']
            )

        ############## Making Predictions ###############
        trainer = pl.Trainer(accelerator=device.type)

        ########## Making predictions for Roq ###########
        # Validation
        ypreds = trainer.predict(model, val_loader)
        ypreds_arr = torch.vstack(ypreds).numpy()

        ypreds_mc = trainer.predict(LitMCDPModel, val_loader)
        ypreds_mc = torch.hstack([torch.Tensor(ypreds_mc[b]) for b in range(len(ypreds_mc))])

        ypreds_val_m, ypreds_val_Ud, ypreds_val_Um = comput_uncertainty(ypreds_mc)

        # Test
        ypreds_test = trainer.predict(model, test_loader)
        ypreds_arr_test = torch.vstack(ypreds_test).numpy()
        ypreds_tens_test_org=yTransFunc.inverse_transform(torch.Tensor(ypreds_arr_test))

        ypreds_test_mc = trainer.predict(LitMCDPModel, test_loader)
        ypreds_test_mc = torch.hstack([torch.Tensor(ypreds_test_mc[b]) for b in range(len(ypreds_test_mc))])

        ypreds_test_m, ypreds_test_Ud, ypreds_test_Um = comput_uncertainty(ypreds_test_mc)
        ypreds_test_m_org=yTransFunc.inverse_transform(torch.Tensor(ypreds_test_m))

        ########## Make predication for Neo and Bao ##########

        # Replace the transformed targets with ones suitable for vanilla models
        yTransFunc = target_transform(train_set)
        train_set = yTransFunc.transform(train_set)
        val_set = yTransFunc.transform(val_set)
        test_set = yTransFunc.transform(test_set)

        val_loader = DataLoader(
            val_set, batch_size=batch_size,
            shuffle=False, num_workers=0, follow_batch=follow_batch,
            #  persistent_workers=True
            )
        test_loader = DataLoader(
            test_set, batch_size=batch_size,
            shuffle=False, num_workers=0, follow_batch=follow_batch,
            #  persistent_workers=True
            )
        
        # Model Inference
        # Neo
        ypreds_test_neo_vanilla = trainer.predict(neo_vanilla, test_loader)
        neo_preds2_vanilla = torch.vstack(ypreds_test_neo_vanilla).numpy()
        neo_preds2_org_vanilla=yTransFunc.inverse_transform(torch.Tensor(neo_preds2_vanilla))

        # Bao
        ypreds_test_bao_vanilla = trainer.predict(bao_vanilla, test_loader)
        bao_preds2_vanilla = torch.vstack(ypreds_test_bao_vanilla).numpy()
        bao_preds2_org_vanilla=yTransFunc.inverse_transform(torch.Tensor(bao_preds2_vanilla))

        
        ############ Tuning strategy parameters ############
        
        # Tune plan selection arguments
        # Thresholds for the Conservative strategy

        cons_total_ratio_RN = find_optimal_cons_ratio(
            ypreds_val_m,ypreds_val_Ud+ypreds_val_Um,
            val_set,
            minimize = 'runtime',
            ratios = np.arange(0.0,2,0.05)
            )

        cons_total_ratio_RN={'cons_ratio':cons_total_ratio_RN}

        cons_total_prune_ratio_RN = find_optimal_prune_ratio(
            'data_prune_ratio', 'model_prune_ratio',
            ypreds_val_m,[ypreds_val_Ud,ypreds_val_Um],
            val_set,
            strategy='conservative',
            minimize = 'runtime',
            p1_grid = np.arange(0,.5,0.1),
            p2_grid = np.arange(0,.5,0.1),
            strategy_args=cons_total_ratio_RN
            )

        cons_total_ratio_SO = find_optimal_cons_ratio(
            ypreds_val_m,ypreds_val_Ud+ypreds_val_Um,
            val_set,
            minimize = 'subopt',
            ratios = np.arange(0.0,2,0.05)
            )
        
        cons_total_ratio_SO={'cons_ratio':cons_total_ratio_SO}

        cons_total_prune_ratio_SO = find_optimal_prune_ratio(
            'data_prune_ratio', 'model_prune_ratio',
            ypreds_val_m,[ypreds_val_Ud,ypreds_val_Um],
            val_set,
            strategy='conservative',
            minimize = 'subopt',
            p1_grid = np.arange(0,.5,0.1),
            p2_grid = np.arange(0,.5,0.1),
            strategy_args=cons_total_ratio_SO
            )

        cons_data_ratio_RN = find_optimal_cons_ratio(
            ypreds_val_m,ypreds_val_Ud,
            val_set,
            minimize = 'runtime',
            ratios = np.arange(0.0,2,0.05)
            )
        
        cons_data_ratio_RN={'cons_ratio':cons_data_ratio_RN}

        cons_data_prune_ratio_RN = find_optimal_prune_ratio(
            'data_prune_ratio',
            'model_prune_ratio',ypreds_val_m,ypreds_val_Ud,
            val_set,
            strategy='conservative',
            minimize = 'runtime',
            p1_grid = np.arange(0,.5,0.1),
            strategy_args=cons_data_ratio_RN
            )

        cons_data_ratio_SO = find_optimal_cons_ratio(
            ypreds_val_m,ypreds_val_Ud,
            val_set,
            minimize = 'subopt',
            ratios = np.arange(0.0,2,0.05)
            )
        
        cons_data_ratio_SO={'cons_ratio':cons_data_ratio_SO}

        cons_data_prune_ratio_SO = find_optimal_prune_ratio(
            'data_prune_ratio',
            'model_prune_ratio',ypreds_val_m,ypreds_val_Ud,
            val_set,
            strategy='conservative',
            minimize = 'subopt',
            p1_grid = np.arange(0,.5,0.1),
            strategy_args=cons_data_ratio_SO
            )

        cons_model_ratio_RN = find_optimal_cons_ratio(
            ypreds_val_m,ypreds_val_Um,
            val_set,
            minimize = 'runtime',
            ratios = np.arange(0.0,2,0.05))
        
        cons_model_ratio_RN={'cons_ratio':cons_model_ratio_RN}

        cons_model_prune_ratio_RN = find_optimal_prune_ratio(
            'data_prune_ratio',
            'model_prune_ratio',ypreds_val_m,ypreds_val_Um,
            val_set,
            strategy='conservative',
            minimize = 'runtime',
            p2_grid = np.arange(0,.5,0.1),
            strategy_args=cons_model_ratio_RN)

        cons_model_ratio_SO = find_optimal_cons_ratio(
            ypreds_val_m,ypreds_val_Um,
            val_set,
            minimize = 'subopt',
            ratios = np.arange(0.0,2,0.05))
        
        cons_model_ratio_SO={'cons_ratio':cons_model_ratio_SO}

        cons_model_prune_ratio_SO = find_optimal_prune_ratio(
            'data_prune_ratio',
            'model_prune_ratio',ypreds_val_m,ypreds_val_Um,
            val_set,
            strategy='conservative',
            minimize = 'subopt',
            p2_grid = np.arange(0,.5,0.1),
            strategy_args=cons_model_ratio_SO
            )

        # Thresholds for the Risk strategy
        risk_total_prune_ratio_RN = find_optimal_prune_ratio(
            'data_prune_ratio',
            'model_prune_ratio',ypreds_val_m,[ypreds_val_Ud,ypreds_val_Um],
            val_set,
            strategy='subopt risk',
            minimize = 'runtime',
            p1_grid = np.arange(0,.5,0.1),
            p2_grid = np.arange(0,.5,0.1))

        risk_total_prune_ratio_SO = find_optimal_prune_ratio(
            'data_prune_ratio',
            'model_prune_ratio',ypreds_val_m,[ypreds_val_Ud,ypreds_val_Um],
            val_set,
            strategy='subopt risk',
            minimize = 'subopt',
            p1_grid = np.arange(0,.5,0.1),
            p2_grid = np.arange(0,.5,0.1)
            )

        risk_data_prune_ratio_RN = find_optimal_prune_ratio(
            'data_prune_ratio',
            'model_prune_ratio',ypreds_val_m,ypreds_val_Ud,
            val_set,
            strategy='subopt risk',
            minimize = 'runtime',
            p1_grid = np.arange(0,.5,0.1))

        risk_data_prune_ratio_SO = find_optimal_prune_ratio(
            'data_prune_ratio',
            'model_prune_ratio',ypreds_val_m,ypreds_val_Ud,
            val_set,
            strategy='subopt risk',
            minimize = 'subopt',
            p1_grid = np.arange(0,.5,0.1))

        risk_model_prune_ratio_RN = find_optimal_prune_ratio(
            'data_prune_ratio',
            'model_prune_ratio',ypreds_val_m,ypreds_val_Um,
            val_set,
            strategy='subopt risk',
            minimize = 'runtime',
            p2_grid = np.arange(0,.5,0.1)
            )

        risk_model_prune_ratio_SO = find_optimal_prune_ratio(
            'data_prune_ratio',
            'model_prune_ratio',ypreds_val_m,ypreds_val_Um,
            val_set,
            strategy='subopt risk',
            minimize = 'subopt',
            p2_grid = np.arange(0,.5,0.1)
            )

        # Threshold for the Optimizer
        opt_data_prune_ratio = find_optimal_prune_ratio(
            'data_prune_ratio', 'model_prune_ratio',
            val_set.opt_cost, ypreds_val_Ud,
            val_set,
            strategy='baseline opt',
            minimize = 'runtime',
            p1_grid = np.arange(0,.5,0.05)
            )
        
            
        ######### Evaluate the models on test data ##########

        strategy_args = risk_total_prune_ratio_RN.copy() # set as default

        # Base Model
        test_base_ml_subopts,test_base_ml_runtimes,test_best_runtimes = evaluate_method(
            ypreds_arr_test[:,0],ypreds_arr_test[:,1],test_set , 
            strategy='baseline ml',)

        test_base_ml_prune_subopts,test_base_ml_prune_runtimes,_ = evaluate_method(
            ypreds_arr_test[:,0],ypreds_arr_test[:,1],test_set , 
            strategy='baseline ml', prune=True,**strategy_args)

        # MC Dropout
        test_mc_ml_subopts,test_mc_ml_runtimes,_ = evaluate_method(
            ypreds_test_m,[ypreds_test_Ud,ypreds_test_Um],test_set , 
            strategy='baseline ml',)

        test_mc_ml_prune_subopts,test_mc_ml_prune_runtimes,_ = evaluate_method(
            ypreds_test_m,[ypreds_test_Ud,ypreds_test_Um],test_set , 
            strategy='baseline ml', prune=True,**strategy_args)

        # Conservative
        test_cons_subopts, test_cons_runtimes,_ = evaluate_method(
            ypreds_test_m,[ypreds_test_Ud,ypreds_test_Um],test_set , 
            strategy='conservative',**cons_total_ratio_RN)

        test_cons_data_subopts, test_cons_data_runtimes,_ = evaluate_method(
            ypreds_test_m,ypreds_test_Ud,test_set , 
            strategy='conservative',**cons_data_ratio_RN)

        test_cons_model_subopts, test_cons_model_runtimes,_ = evaluate_method(
            ypreds_test_m,ypreds_test_Um,test_set , 
            strategy='conservative',**cons_model_ratio_RN)

        test_cons_prune_RN_subopts, test_cons_prune_RN_runtimes,_ = evaluate_method(
            ypreds_test_m,[ypreds_test_Ud,ypreds_test_Um],test_set , 
            strategy='conservative', prune=True, **cons_total_prune_ratio_RN)

        test_cons_prune_SO_subopts, test_cons_prune_SO_runtimes,_ = evaluate_method(
            ypreds_test_m,[ypreds_test_Ud,ypreds_test_Um],test_set , 
            strategy='conservative', prune=True, **cons_total_prune_ratio_SO)

        test_cons_data_prune_RN_subopts, test_cons_data_prune_RN_runtimes,_ = evaluate_method(
            ypreds_test_m,ypreds_test_Ud,test_set , 
            strategy='conservative', prune=True, **cons_data_prune_ratio_RN)

        test_cons_data_prune_SO_subopts, test_cons_data_prune_SO_runtimes,_ = evaluate_method(
            ypreds_test_m,ypreds_test_Ud,test_set , 
            strategy='conservative', prune=True, **cons_data_prune_ratio_SO)

        test_cons_model_prune_RN_subopts, test_cons_model_prune_RN_runtimes,_ = evaluate_method(
            ypreds_test_m,ypreds_test_Um,test_set , 
            strategy='conservative', prune=True, **cons_model_prune_ratio_RN)

        test_cons_model_prune_SO_subopts, test_cons_model_prune_SO_runtimes,_ = evaluate_method(
            ypreds_test_m,ypreds_test_Um,test_set , 
            strategy='conservative', prune=True, **cons_model_prune_ratio_SO)

        # Risk
        test_risk_subopts, test_risk_runtimes,_ = evaluate_method(
            ypreds_test_m,[ypreds_test_Ud,ypreds_test_Um], test_set , 
            strategy='subopt risk')

        test_risk_data_subopts, test_risk_data_runtimes,_= evaluate_method(
            ypreds_test_m,ypreds_test_Ud, test_set, 
            strategy='subopt risk')

        test_risk_model_subopts, test_risk_model_runtimes,_= evaluate_method(
            ypreds_test_m,ypreds_test_Um, test_set, 
            strategy='subopt risk')

        test_risk_prune_RN_subopts, test_risk_prune_RN_runtimes,_ = evaluate_method(
            ypreds_test_m,[ypreds_test_Ud,ypreds_test_Um], test_set ,  
            strategy='subopt risk',prune=True,**risk_total_prune_ratio_RN)

        test_risk_prune_SO_subopts, test_risk_prune_SO_runtimes,_ = evaluate_method(
            ypreds_test_m,[ypreds_test_Ud,ypreds_test_Um], test_set ,  
            strategy='subopt risk',prune=True,**risk_total_prune_ratio_SO)

        test_risk_data_prune_RN_subopts, test_risk_data_prune_RN_runtimes,_ = evaluate_method(
            ypreds_test_m,ypreds_test_Ud, test_set ,  
            strategy='subopt risk',prune=True,**risk_data_prune_ratio_RN)

        test_risk_data_prune_SO_subopts, test_risk_data_prune_SO_runtimes,_ = evaluate_method(
            ypreds_test_m,ypreds_test_Ud, test_set ,  
            strategy='subopt risk',prune=True,**risk_data_prune_ratio_SO)

        test_risk_model_prune_RN_subopts, test_risk_model_prune_RN_runtimes,_ = evaluate_method(
            ypreds_test_m,ypreds_test_Ud, test_set ,  
            strategy='subopt risk',prune=True,**risk_model_prune_ratio_RN)

        test_risk_model_prune_SO_subopts, test_risk_model_prune_SO_runtimes,_ = evaluate_method(
            ypreds_test_m,ypreds_test_Um, test_set ,  
            strategy='subopt risk',prune=True,**risk_model_prune_ratio_SO)

        # Optimizer
        test_opt_subopts, test_opt_runtimes,_ = evaluate_method(
            ypreds_test_m,ypreds_test_Ud,test_set , 
            strategy='baseline opt')

        test_opt_prune_subopts, test_opt_prune_runtimes,_ = evaluate_method(
            ypreds_test_m,ypreds_test_Ud,test_set , 
            strategy='baseline opt',prune=True,**opt_data_prune_ratio)
        
        # Neo and Bao Vanilla
        neo_subopt2_vanilla,neo_runtime2_vanilla,_ = evaluate_method(
            neo_preds2_vanilla[:,0],neo_preds2_vanilla[:,1],test_set , 
            strategy='baseline ml')

        bao_subopt2_vanilla,bao_runtime2_vanilla,_ = evaluate_method(
            bao_preds2_vanilla[:,0],bao_preds2_vanilla[:,1],test_set , 
            strategy='baseline ml')


        ################## Compute q-error ##################
        q_error_num_joins, _ = lcm_model_eval(
            test_set, ypreds_tens_test_org[:,0], 
            dataset_label = 'Test data', model_label = 'roq',
            load_from_disk=False, files_id = files_id,show_fig=show_fig
            )
        
        roq_mcdp_num_joins, _ = lcm_model_eval(
            test_set, ypreds_test_m_org, 
            dataset_label = 'Test data', model_label = 'roq',
            load_from_disk=False, files_id = files_id,show_fig=show_fig
            )
        
        neo_qerror2_vanilla, _ = lcm_model_eval(
            test_set, neo_preds2_org_vanilla[:,0], 
            dataset_label = 'Test data', model_label = 'neo',
            load_from_disk=False, files_id = files_id,show_fig=show_fig
            )
        
        bao_qerror2_vanilla, _ = lcm_model_eval(
            test_set, bao_preds2_org_vanilla[:,0], 
            dataset_label = 'Test data', model_label = 'bao'
            ,load_from_disk=False, files_id = files_id,show_fig=show_fig
            )
        
        qerrors = pd.DataFrame(
            [bao_qerror2_vanilla.q_error,
             neo_qerror2_vanilla.q_error,
            q_error_num_joins.q_error,
            roq_mcdp_num_joins.q_error
            ]).T
        qerrors.columns = ['Bao', 'Neo','Roq','Roq MCDO']
    

        ########### Compute Perason's correlations ############
        target = test_set.y.numpy()
        pearson_coef1, _ = compute_pearsonr(test_set.opt_cost,target)
        pearson_coef4, _ = compute_pearsonr(ypreds_tens_test_org[:,0].squeeze(),target)
        pearson_coef5, _ = compute_pearsonr(bao_preds2_org_vanilla[:,0].squeeze(),target)
        pearson_coef6, _ = compute_pearsonr(neo_preds2_org_vanilla[:,0].squeeze(),target)
        pearson_coef7, _ = compute_pearsonr(ypreds_test_m_org,target)

        ########### Compute Spearman's correlations ###########
        spearmanr_coef1, _ = compute_spearmanr((test_set.opt_cost),target)
        spearmanr_coef4, _ = compute_spearmanr(ypreds_tens_test_org[:,0].squeeze(),target)
        spearmanr_coef5, _ = compute_spearmanr(bao_preds2_org_vanilla[:,0].squeeze(),target)
        spearmanr_coef6, _ = compute_spearmanr(neo_preds2_org_vanilla[:,0].squeeze(),target)
        spearmanr_coef7, _ = compute_spearmanr(ypreds_test_m_org,target)

        labels = ['Cost','Bao', 'Neo','Roq','Roq MCDO']
        
        pearson_values = [
            pearson_coef1,pearson_coef5,pearson_coef6,
            pearson_coef4,pearson_coef7
            ]
        
        spearman_values = [
            spearmanr_coef1,spearmanr_coef5,spearmanr_coef6,
            spearmanr_coef4,spearmanr_coef7
            ]
        
        corr_df = pd.DataFrame(
            np.array([pearson_values,spearman_values]), 
            columns=labels,index=['Pearson','Spearman']
            )

        labels = [
            'optimizer',
            'optimizer pruned',
            'Neo','Bao',
            'base model',
            'base model pruned',
            'mc dropout',
            'mc dropout pruned',
            'conservative',
            'cons. data unc.',
            'cons. model unc.',
            'cons. prun. RN',
            'cons. prun. SO',
            'cons. prun. data unc. RN',
            'cons. prun. data unc. SO',
            'cons. prun. model unc. RN',
            'cons. prun. model unc. SO',
            'risk',
            'risk data unc.',
            'risk model unc.',
            'risk prun. RN',
            'risk prun. SO',
            'risk prun. data unc. RN',
            'risk prun. data unc. SO',
            'risk prun. model unc. RN',
            'risk prun. model unc. SO',
            'best'
            ]
        
        rt_values = [
            test_opt_runtimes,
            test_opt_prune_runtimes,
            neo_runtime2_vanilla,bao_runtime2_vanilla,
            test_base_ml_runtimes,
            test_base_ml_prune_runtimes,
            test_mc_ml_runtimes,
            test_mc_ml_prune_runtimes,
            test_cons_runtimes,
            test_cons_data_runtimes,
            test_cons_model_runtimes,
            test_cons_prune_RN_runtimes,
            test_cons_prune_SO_runtimes,
            test_cons_data_prune_RN_runtimes,
            test_cons_data_prune_SO_runtimes,
            test_cons_model_prune_RN_runtimes,
            test_cons_model_prune_SO_runtimes,
            test_risk_runtimes, 
            test_risk_data_runtimes,
            test_risk_model_runtimes,
            test_risk_prune_RN_runtimes,
            test_risk_prune_SO_runtimes,
            test_risk_data_prune_RN_runtimes,
            test_risk_data_prune_SO_runtimes,
            test_risk_model_prune_RN_runtimes,
            test_risk_model_prune_SO_runtimes,
            test_best_runtimes
            ]
        
        so_values = [
            test_opt_subopts,
            test_opt_prune_subopts,
            neo_subopt2_vanilla,bao_subopt2_vanilla,
            test_base_ml_subopts,
            test_base_ml_prune_subopts,
            test_mc_ml_subopts,
            test_mc_ml_prune_subopts,
            test_cons_subopts,
            test_cons_data_subopts,
            test_cons_model_subopts,
            test_cons_prune_RN_subopts,
            test_cons_prune_SO_subopts,
            test_cons_data_prune_RN_subopts,
            test_cons_data_prune_SO_subopts,
            test_cons_model_prune_RN_subopts,
            test_cons_model_prune_SO_subopts,
            test_risk_subopts,
            test_risk_data_subopts,
            test_risk_model_subopts,
            test_risk_prune_RN_subopts,
            test_risk_prune_SO_subopts,
            test_risk_data_prune_RN_subopts,
            test_risk_data_prune_SO_subopts,
            test_risk_model_prune_RN_subopts,
            test_risk_model_prune_SO_subopts,
            ]


        runtime_details = pd.DataFrame(
            np.array(rt_values).transpose(),
            columns=labels
            )
        
        runtime_res = compare_with_baseline(rt_values, labels).sort_values(by='total_runtime_change%')

        so_labels = [i for i in labels if i not in ['best']]

        subopt_details = pd.DataFrame(np.array(so_values).transpose(),
                                        columns=so_labels)
        
        subopt_res = subopt_details.describe(
                        percentiles=[.5,.6,.7,.8,.90,.95,.99]).T
        subopt_res = subopt_res.loc[:,['mean','50%','60%','70%','80%','90%','95%','99%','max']].sort_values(by='mean')

        agg_res = runtime_res.merge(subopt_res,how='outer',left_index=True,right_index=True)
        #runtime_res.join(subopt_res).sort_values(by='mean')
        
        qerror_dict[i]=qerrors
        corr_dict[i]=corr_df
        rt_res_dict[i]=runtime_details
        so_res_dict[i]=subopt_details
        agg_res_dict[i]=agg_res



    with open('./results/qerror_dict_{}.pkl'.format(experiment_id), 'wb') as file:
        pickle.dump(qerror_dict, file)

    with open('./results/corr_dict_{}.pkl'.format(experiment_id), 'wb') as file:
        pickle.dump(corr_dict, file)

    with open('./results/rt_res_dict_{}.pkl'.format(experiment_id), 'wb') as file:
        pickle.dump(rt_res_dict, file)

    with open('./results/so_res_dict_{}.pkl'.format(experiment_id), 'wb') as file:
        pickle.dump(so_res_dict, file)

    with open('./results/agg_res_dict_{}.pkl'.format(experiment_id), 'wb') as file:
        pickle.dump(agg_res_dict, file)

if __name__ == '__main__':

    test(
        experiment_id = 'job',
        architecture_p = 'roq',
        files_id = 'job_syn_p4',
        try_gpu_inf=True,
        seed = 0, 
        num_workers=num_workers,
        show_fig = False
        )