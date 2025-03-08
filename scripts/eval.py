####################### TEST ROQ #######################

import os
import pickle
import torch
import numpy as np
import pandas as pd
from util.pyg_data import queryPlanPGDataset_withbenchmark
from util.util import set_seed, load_model_params, load_best_model_paths
from util.data_transform import *
from util.custom_loss import aleatoric_loss, rmse_loss, bce_loss
import pytorch_lightning as pl
from lcm.roq_model import lcm_pl as roq
from lcm.neo_bao_model import lcm_pl as neo_bao
from lcm.lero_model import LeroModelPairwise as leroPairwise
from lcm.balsa_model import balsa_model
from torch_geometric.loader import DataLoader
from util.torch_util import LitMCdropoutModel
from util.eval_util import *


results_dir = os.path.join('.','results')
   
def test(
        experiment_id = 'job',
        file_experiment_id = None,
        files_id = 'temp', proc_files_id=None, try_gpu_inf=True, 
        seed = 1, num_workers=5, show_fig = False,
        benchmark_files_id='',
        labeled_data_dir = './labeled_data/',
        test_split = 'test',
        cons_ratio_sp = np.arange(0.0,3,0.1),
        prune_ratio_sp = np.arange(0,.75,0.05),
        reload_data = True,
        val_samples = 500,
        test_samples = 500,
        test_slow_samples = None
        ):

    set_seed(seed)

    if file_experiment_id is None:
        file_experiment_id = experiment_id

    # setting torch.device()
    if try_gpu_inf:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    if try_gpu_inf and str(device) != 'cuda':
        print("Warning: GPU was not available for inferenec. Using CPU instead.")
    print("Device used for inference:",str(device).upper())

    # create results dir if it does not exist
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # loading model hyper-parameters
    for benchmark in ['ceb','dsb','job']:
        if benchmark in experiment_id:
            config_file=f'model_params_{benchmark}.cfg'
            break
    
    roq_config = load_model_params('roq',config_file=config_file)
    neo_config = load_model_params('neo',config_file=config_file)
    bao_config = load_model_params('bao',config_file=config_file)
    lero_config = load_model_params('lero',config_file=config_file)
    balsa_config = load_model_params('balsa',config_file=config_file)

    # drop training hyper-parameters from the configs
    configs = [roq_config,neo_config,bao_config,lero_config,balsa_config]
    params_to_drop = ['max_epochs','patience','min_delta']
    for config in configs:
        for param in params_to_drop:
            if param in config:
                config.pop(param)

    # loading model paths
    best_model_paths = load_best_model_paths('roq',experiment_id)
    neo_paths =  load_best_model_paths('neo',experiment_id)
    bao_paths =  load_best_model_paths('bao',experiment_id)
    lero_paths = load_best_model_paths('lero',experiment_id)
    balsa_paths = load_best_model_paths('balsa',experiment_id)

    # Load train, validation, and test datasets
    print("loading train")
    train_set = queryPlanPGDataset_withbenchmark(
        split= 'train', files_id = files_id,
        proc_files_id = proc_files_id,
        benchmark_files_id=benchmark_files_id,
        labeled_data_dir=labeled_data_dir,  
        force_reload=reload_data, seed = seed,
        val_samples = val_samples, test_samples = test_samples,
        test_slow_samples = test_slow_samples,
        exp_id=file_experiment_id
        )
    print("{} queries and {} samples in training dataset: ".format(np.unique(np.array(train_set.query_id)).shape[0],train_set.len()))
    
    print("loading val")
    val_set = queryPlanPGDataset_withbenchmark(
        split= 'val', 
        files_id = files_id,
        proc_files_id=proc_files_id,
        exp_id=file_experiment_id
        )
    print("{} queries and {} samples in vlidation dataset: ".format(np.unique(np.array(val_set.query_id)).shape[0],val_set.len()))
    
    print("loading test")
    test_set = queryPlanPGDataset_withbenchmark(
        split= test_split, 
        files_id = files_id,
        proc_files_id=proc_files_id,
        exp_id=file_experiment_id
        )
    print("{} queries and {} samples in test dataset: ".format(np.unique(np.array(test_set.query_id)).shape[0],test_set.len()))
    
    # Perform data transformations on inputs 
    null_imp = nullImputation(train_set)
    train_set = null_imp(train_set)
    val_set = null_imp(val_set)
    test_set = null_imp(test_set)

    minmax_scale = minmaxScale(train_set)
    train_set = minmax_scale(train_set)
    val_set = minmax_scale(val_set)
    test_set = minmax_scale(test_set)

    # Initialize data transformations on targets 
    yTransFunc_log = target_log_transform(train_set)
    # yTransFunc = target_transform(train_set)

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
    bce_l = bce_loss()
    rmse_l = rmse_loss()

    torch.set_float32_matmul_precision('high')

    qerror_dict = {}
    corr_dict = {}
    rt_res_dict = {}
    so_res_dict = {}
    agg_res_dict = {}

    # For loop starts here
    max_num_models = max(len(best_model_paths),len(neo_paths),len(bao_paths),len(lero_paths),len(balsa_paths))
    for i in range(max_num_models):
        # change seed for each iteration
        seed+=1
        set_seed(seed)

        # get model paths
        best_model_path = best_model_paths[i] if i < len(best_model_paths) else best_model_paths[-1]
        neo_path = neo_paths[i] if i < len(neo_paths) else neo_paths[-1]
        bao_path = bao_paths[i] if i < len(bao_paths) else bao_paths[-1]
        lero_path = lero_paths[i] if i < len(lero_paths) else lero_paths[-1]
        balsa_path = balsa_paths[i] if i < len(balsa_paths) else balsa_paths[-1]

        # Note: model paramteres must be passed unless they are the same as defaults
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

        lero_pairwise = leroPairwise.load_from_checkpoint(
            lero_path,
            num_node = node_attr_shape[0], 
            numPlanFeat=plan_attr_shape,
            numPlanOrdFeat=plan_ord_shape,
            numQueryGraphFeat = graph_attr_shape[0],
            device = device, 
            criterion = rmse_l,
            **lero_config
            )
        lero=lero_pairwise.lero_model

        pretrain_balsa = balsa_config.pop('pretrain',False)
        balsa = balsa_model.load_from_checkpoint(
            balsa_path,
            num_node = node_attr_shape[0], 
            node_dim = node_attr_shape[1],
            edge_dim = edge_attr_shape[1],#fill_value =0, 
            numPlanFeat=plan_attr_shape,
            numPlanOrdFeat=plan_ord_shape,
            numQueryGraphFeat = graph_attr_shape[0],
            device = device, 
            criterion = rmse_l,
            **balsa_config
            )

        

        # Transform targets for Roq
        train_set = yTransFunc_log.transform(train_set)
        val_set = yTransFunc_log.transform(val_set)
        test_set = yTransFunc_log.transform(test_set)

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
        ypreds_tens_test_org=yTransFunc_log.inverse_transform(torch.Tensor(ypreds_arr_test))

        ypreds_test_mc = trainer.predict(LitMCDPModel, test_loader)
        ypreds_test_mc = torch.hstack([torch.Tensor(ypreds_test_mc[b]) for b in range(len(ypreds_test_mc))])

        ypreds_test_m, ypreds_test_Ud, ypreds_test_Um = comput_uncertainty(ypreds_test_mc)
        ypreds_test_m_org=yTransFunc_log.inverse_transform(torch.Tensor(ypreds_test_m))

        ########## Make predication for Neo and Bao ##########
     
        # Model Inference
        # Neo
        ypreds_test_neo_vanilla = trainer.predict(neo_vanilla, test_loader)
        neo_preds2_vanilla = torch.vstack(ypreds_test_neo_vanilla).numpy()
        neo_preds2_org_vanilla=yTransFunc_log.inverse_transform(torch.Tensor(neo_preds2_vanilla))

        # Bao
        ypreds_test_bao_vanilla = trainer.predict(bao_vanilla, test_loader)
        bao_preds2_vanilla = torch.vstack(ypreds_test_bao_vanilla).numpy()
        bao_preds2_org_vanilla=yTransFunc_log.inverse_transform(torch.Tensor(bao_preds2_vanilla))

        # Lero
        ypreds_test_lero = trainer.predict(lero, test_loader)
        ypreds_test_lero = torch.vstack(ypreds_test_lero)
        # ypreds_test_lero=yTransFunc.inverse_transform(torch.Tensor(ypreds_test_lero))

        # Balsa
        ypreds_test_balsa = trainer.predict(balsa, test_loader)
        ypreds_test_balsa = torch.vstack(ypreds_test_balsa)
        ypreds_test_balsa=yTransFunc_log.inverse_transform(torch.Tensor(ypreds_test_balsa))
        
        ############ Tuning strategy parameters ############
        
        # Tune plan selection arguments
        # Thresholds for the Conservative strategy

        cons_total_ratio_RN = find_optimal_cons_ratio(
            ypreds_val_m,ypreds_val_Ud+ypreds_val_Um,
            val_set,
            minimize = 'runtime',
            ratios = cons_ratio_sp
            )

        cons_total_ratio_RN={'cons_ratio':cons_total_ratio_RN}

        cons_total_prune_ratio_RN = find_optimal_prune_ratio(
            'data_prune_ratio', 'model_prune_ratio',
            ypreds_val_m,[ypreds_val_Ud,ypreds_val_Um],
            val_set,
            strategy='conservative',
            minimize = 'runtime',
            p1_grid = prune_ratio_sp,
            p2_grid = prune_ratio_sp,
            strategy_args=cons_total_ratio_RN
            )

        cons_total_ratio_SO = find_optimal_cons_ratio(
            ypreds_val_m,ypreds_val_Ud+ypreds_val_Um,
            val_set,
            minimize = 'subopt',
            ratios = cons_ratio_sp
            )
        
        cons_total_ratio_SO={'cons_ratio':cons_total_ratio_SO}

        cons_total_prune_ratio_SO = find_optimal_prune_ratio(
            'data_prune_ratio', 'model_prune_ratio',
            ypreds_val_m,[ypreds_val_Ud,ypreds_val_Um],
            val_set,
            strategy='conservative',
            minimize = 'subopt',
            p1_grid = prune_ratio_sp,
            p2_grid = prune_ratio_sp,
            strategy_args=cons_total_ratio_SO
            )

        cons_data_ratio_RN = find_optimal_cons_ratio(
            ypreds_val_m,ypreds_val_Ud,
            val_set,
            minimize = 'runtime',
            ratios = cons_ratio_sp
            )
        
        cons_data_ratio_RN={'cons_ratio':cons_data_ratio_RN}

        cons_data_prune_ratio_RN = find_optimal_prune_ratio(
            'data_prune_ratio',
            'model_prune_ratio',ypreds_val_m,ypreds_val_Ud,
            val_set,
            strategy='conservative',
            minimize = 'runtime',
            p1_grid = prune_ratio_sp,
            strategy_args=cons_data_ratio_RN
            )

        cons_data_ratio_SO = find_optimal_cons_ratio(
            ypreds_val_m,ypreds_val_Ud,
            val_set,
            minimize = 'subopt',
            ratios = cons_ratio_sp
            )
        
        cons_data_ratio_SO={'cons_ratio':cons_data_ratio_SO}

        cons_data_prune_ratio_SO = find_optimal_prune_ratio(
            'data_prune_ratio',
            'model_prune_ratio',ypreds_val_m,ypreds_val_Ud,
            val_set,
            strategy='conservative',
            minimize = 'subopt',
            p1_grid = prune_ratio_sp,
            strategy_args=cons_data_ratio_SO
            )

        cons_model_ratio_RN = find_optimal_cons_ratio(
            ypreds_val_m,ypreds_val_Um,
            val_set,
            minimize = 'runtime',
            ratios = cons_ratio_sp
            )
        
        cons_model_ratio_RN={'cons_ratio':cons_model_ratio_RN}

        cons_model_prune_ratio_RN = find_optimal_prune_ratio(
            'data_prune_ratio',
            'model_prune_ratio',ypreds_val_m,ypreds_val_Um,
            val_set,
            strategy='conservative',
            minimize = 'runtime',
            p2_grid = prune_ratio_sp,
            strategy_args=cons_model_ratio_RN)

        cons_model_ratio_SO = find_optimal_cons_ratio(
            ypreds_val_m,ypreds_val_Um,
            val_set,
            minimize = 'subopt',
            ratios = cons_ratio_sp
            )
        
        cons_model_ratio_SO={'cons_ratio':cons_model_ratio_SO}

        cons_model_prune_ratio_SO = find_optimal_prune_ratio(
            'data_prune_ratio',
            'model_prune_ratio',ypreds_val_m,ypreds_val_Um,
            val_set,
            strategy='conservative',
            minimize = 'subopt',
            p2_grid = prune_ratio_sp,
            strategy_args=cons_model_ratio_SO
            )

        # Thresholds for the Risk strategy
        risk_total_prune_ratio_RN = find_optimal_prune_ratio(
            'data_prune_ratio',
            'model_prune_ratio',ypreds_val_m,[ypreds_val_Ud,ypreds_val_Um],
            val_set,
            strategy='subopt risk',
            minimize = 'runtime',
            p1_grid = prune_ratio_sp,
            p2_grid = prune_ratio_sp)

        risk_total_prune_ratio_SO = find_optimal_prune_ratio(
            'data_prune_ratio',
            'model_prune_ratio',ypreds_val_m,[ypreds_val_Ud,ypreds_val_Um],
            val_set,
            strategy='subopt risk',
            minimize = 'subopt',
            p1_grid = prune_ratio_sp,
            p2_grid = prune_ratio_sp
            )

        risk_data_prune_ratio_RN = find_optimal_prune_ratio(
            'data_prune_ratio',
            'model_prune_ratio',ypreds_val_m,ypreds_val_Ud,
            val_set,
            strategy='subopt risk',
            minimize = 'runtime',
            p1_grid = prune_ratio_sp)

        risk_data_prune_ratio_SO = find_optimal_prune_ratio(
            'data_prune_ratio',
            'model_prune_ratio',ypreds_val_m,ypreds_val_Ud,
            val_set,
            strategy='subopt risk',
            minimize = 'subopt',
            p1_grid = prune_ratio_sp)

        risk_model_prune_ratio_RN = find_optimal_prune_ratio(
            'data_prune_ratio',
            'model_prune_ratio',ypreds_val_m,ypreds_val_Um,
            val_set,
            strategy='subopt risk',
            minimize = 'runtime',
            p2_grid = prune_ratio_sp
            )

        risk_model_prune_ratio_SO = find_optimal_prune_ratio(
            'data_prune_ratio',
            'model_prune_ratio',ypreds_val_m,ypreds_val_Um,
            val_set,
            strategy='subopt risk',
            minimize = 'subopt',
            p2_grid = prune_ratio_sp
            )

        # Threshold for the Optimizer
        opt_data_prune_ratio = find_optimal_prune_ratio(
            'data_prune_ratio', 'model_prune_ratio',
            val_set.opt_cost, ypreds_val_Ud,
            val_set,
            strategy='baseline opt',
            minimize = 'runtime',
            p1_grid = prune_ratio_sp
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

        lero_subopt,lero_runtime,_ = evaluate_method(
            ypreds_test_lero[:,0],ypreds_test_lero[:,1],test_set, 
            strategy='baseline ml')
    
        balsa_subopt,balsa_runtime,_ = evaluate_method(
            ypreds_test_balsa[:,0],ypreds_test_balsa[:,1],test_set, 
            strategy='baseline ml')

        ################## Compute q-error ##################
        q_error_num_joins, _ = lcm_model_eval(
            test_set, ypreds_tens_test_org[:,0], 
            dataset_label = 'test_data', model_label = 'roq',
            load_from_disk=False, files_id = files_id,show_fig=show_fig
            )
        
        roq_mcdp_num_joins, _ = lcm_model_eval(
            test_set, ypreds_test_m_org, 
            dataset_label = 'test_data', model_label = 'roq',
            load_from_disk=False, files_id = files_id,show_fig=show_fig
            )
        
        neo_qerror2_vanilla, _ = lcm_model_eval(
            test_set, neo_preds2_org_vanilla[:,0], 
            dataset_label = 'test_data', model_label = 'neo',
            load_from_disk=False, files_id = files_id,show_fig=show_fig
            )
        
        bao_qerror2_vanilla, _ = lcm_model_eval(
            test_set, bao_preds2_org_vanilla[:,0], 
            dataset_label = 'test_data', model_label = 'bao'
            ,load_from_disk=False, files_id = files_id,show_fig=show_fig
            )
        
        lero_qerror, _ = lcm_model_eval(
            test_set, ypreds_test_lero[:,0], 
            dataset_label = 'test_data', model_label = 'lero'
            ,load_from_disk=False, files_id = files_id,show_fig=show_fig
            )
        
        balsa_qerror, _ = lcm_model_eval(
            test_set, ypreds_test_balsa[:,0], 
            dataset_label = 'test_data', model_label = 'balsa'
            ,load_from_disk=False, files_id = files_id,show_fig=show_fig
            )
        
        qerrors = pd.DataFrame(
            [bao_qerror2_vanilla.q_error,
             neo_qerror2_vanilla.q_error,
            lero_qerror.q_error,
            balsa_qerror.q_error,
            q_error_num_joins.q_error,
            roq_mcdp_num_joins.q_error
            ]).T
        qerrors.columns = ['Bao','Neo','Lero','Balsa','Roq','Roq MCDO']
    

        ########### Compute Perason's correlations ############
        target = test_set.y.numpy()
        pearson_coef1, _ = compute_pearsonr(test_set.opt_cost,target)
        pearson_coef4, _ = compute_pearsonr(ypreds_tens_test_org[:,0].squeeze(),target)
        pearson_coef5, _ = compute_pearsonr(bao_preds2_org_vanilla[:,0].squeeze(),target)
        pearson_coef6, _ = compute_pearsonr(neo_preds2_org_vanilla[:,0].squeeze(),target)
        pearson_coef7, _ = compute_pearsonr(ypreds_test_m_org,target)
        pearson_coef8, _ = compute_pearsonr(ypreds_test_lero[:,0].squeeze(),target)
        pearson_coef9, _ = compute_pearsonr(ypreds_test_balsa[:,0].squeeze(),target)

        ########### Compute Spearman's correlations ###########
        spearmanr_coef1, _ = compute_spearmanr((test_set.opt_cost),target)
        spearmanr_coef4, _ = compute_spearmanr(ypreds_tens_test_org[:,0].squeeze(),target)
        spearmanr_coef5, _ = compute_spearmanr(bao_preds2_org_vanilla[:,0].squeeze(),target)
        spearmanr_coef6, _ = compute_spearmanr(neo_preds2_org_vanilla[:,0].squeeze(),target)
        spearmanr_coef7, _ = compute_spearmanr(ypreds_test_m_org,target)
        spearmanr_coef8, _ = compute_spearmanr(ypreds_test_lero[:,0].squeeze(),target)
        spearmanr_coef9, _ = compute_spearmanr(ypreds_test_balsa[:,0].squeeze(),target)

        pearson_values = [
            pearson_coef1,pearson_coef5,pearson_coef6,
            pearson_coef8,pearson_coef9,pearson_coef4,pearson_coef7
            ]
        
        spearman_values = [
            spearmanr_coef1,spearmanr_coef5,spearmanr_coef6,
            spearmanr_coef8,spearmanr_coef9,spearmanr_coef4,spearmanr_coef7
            ]
        
        model_labels = ['Cost','Bao','Neo','Lero','Balsa','Roq','Roq MCDO']
        corr_df = pd.DataFrame(
            np.array([pearson_values,spearman_values]), 
            columns=model_labels,index=['Pearson','Spearman']
            )

        labels = [
            'optimizer',
            'optimizer pruned',
            'Neo','Bao', 'Lero', 'Balsa',
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
            lero_runtime, balsa_runtime,
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
            lero_subopt, balsa_subopt,
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

        q_ids = np.unique(np.array(test_set.query_id))
        runtime_details = pd.DataFrame(
            np.array(rt_values).transpose(),
            columns=labels,index=q_ids
            )
        
        runtime_res = compare_with_baseline(rt_values, labels).sort_values(by='total_runtime_change%')

        so_labels = [i for i in labels if i not in ['best']]

        subopt_details = pd.DataFrame(
            np.array(so_values).transpose(),
            columns=so_labels,index=q_ids
            )
        
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



    with open(os.path.join(results_dir,'qerror_dict_{}_{}.pkl'.format(experiment_id,test_split)), 'wb') as file:
        pickle.dump(qerror_dict, file)

    with open(os.path.join(results_dir,'corr_dict_{}_{}.pkl'.format(experiment_id,test_split)), 'wb') as file:
        pickle.dump(corr_dict, file)

    with open(os.path.join(results_dir,'rt_res_dict_{}_{}.pkl'.format(experiment_id,test_split)), 'wb') as file:
        pickle.dump(rt_res_dict, file)

    with open(os.path.join(results_dir,'so_res_dict_{}_{}.pkl'.format(experiment_id,test_split)), 'wb') as file:
        pickle.dump(so_res_dict, file)

    with open(os.path.join(results_dir,'agg_res_dict_{}_{}.pkl'.format(experiment_id,test_split)), 'wb') as file:
        pickle.dump(agg_res_dict, file)

if __name__ == '__main__':

    test(
        experiment_id = 'ceb_1000_x5_s317',
        file_experiment_id = 'ceb_1000',
        files_id = 'ceb_1000',
        proc_files_id='ceb_1000',
        benchmark_files_id = 'job_v2.2',
        labeled_data_dir='./labeled_data/ceb/',
        test_split = 'test',
        try_gpu_inf=True,
        num_workers=1,
        show_fig = False,
        cons_ratio_sp = np.arange(0.0,3,0.01),
        prune_ratio_sp = np.arange(0,.3,0.025),
        seed = 310,
        reload_data = False,
        val_samples = 0.1,
        test_samples = 0.1,
        test_slow_samples = 0.5
        )