####################### Single Test #######################

import os
import torch
import numpy as np
from util.util import set_seed, load_model_params
from util.data_transform import *
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from util.torch_util import LitMCdropoutModel
from util.eval_util import *


results_dir = os.path.join('.','results')
   
def test(
    files_id = 'temp', try_gpu_inf=True, 
    seed = 1, num_workers=5, show_fig = False,
    cons_ratio_sp = np.arange(0.0,3,0.1),
    prune_ratio_sp = np.arange(0,.75,0.05),
    models = None,
    val_set = None,
    test_set = None,
    yTransFunc = None
    ):

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
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # loading model hyper-parameters
    roq_config = load_model_params('roq')

    # unpack models
    if models is None or len(models) != 5:
        raise ValueError("The 'models' variable must contain exactly five elements.")
    roq, balsa, neo, bao, lero = models

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

    torch.set_float32_matmul_precision('high')

    # Enables variational inference using MC dropout
    LitMCDPModel = LitMCdropoutModel(
        roq, mc_iteration=10, 
        dropout=roq_config['dropout']
        )

    ############## Making Predictions ###############
    trainer = pl.Trainer(accelerator=device.type)


    ########## Making predictions for Roq ###########
    # Validation
    ypreds = trainer.predict(roq, val_loader)
    ypreds_arr = torch.vstack(ypreds).numpy()

    ypreds_mc = trainer.predict(LitMCDPModel, val_loader)
    ypreds_mc = torch.hstack([torch.Tensor(ypreds_mc[b]) for b in range(len(ypreds_mc))])

    ypreds_val_m, ypreds_val_Ud, ypreds_val_Um = comput_uncertainty(ypreds_mc)

    # Test
    ypreds_test = trainer.predict(roq, test_loader)
    ypreds_arr_test = torch.vstack(ypreds_test).numpy()
    ypreds_tens_test_org=yTransFunc.inverse_transform(torch.Tensor(ypreds_arr_test))

    ypreds_test_mc = trainer.predict(LitMCDPModel, test_loader)
    ypreds_test_mc = torch.hstack([torch.Tensor(ypreds_test_mc[b]) for b in range(len(ypreds_test_mc))])

    ypreds_test_m, ypreds_test_Ud, ypreds_test_Um = comput_uncertainty(ypreds_test_mc)
    ypreds_test_m_org=yTransFunc.inverse_transform(torch.Tensor(ypreds_test_m))

    ######### Make predication for the baselines ##########
    # Model Inference
    # Neo
    ypreds_test_neo_vanilla = trainer.predict(neo, test_loader)
    neo_preds2_vanilla = torch.vstack(ypreds_test_neo_vanilla).numpy()
    neo_preds2_org_vanilla=yTransFunc.inverse_transform(torch.Tensor(neo_preds2_vanilla))

    # Bao
    ypreds_test_bao_vanilla = trainer.predict(bao, test_loader)
    bao_preds2_vanilla = torch.vstack(ypreds_test_bao_vanilla).numpy()
    bao_preds2_org_vanilla=yTransFunc.inverse_transform(torch.Tensor(bao_preds2_vanilla))

    # Lero
    ypreds_test_lero = trainer.predict(lero, test_loader)
    ypreds_test_lero = torch.vstack(ypreds_test_lero)
    # ypreds_test_lero=yTransFunc.inverse_transform(torch.Tensor(ypreds_test_lero))

    # Balsa
    ypreds_test_balsa = trainer.predict(balsa, test_loader)
    ypreds_test_balsa = torch.vstack(ypreds_test_balsa)
    ypreds_test_balsa=yTransFunc.inverse_transform(torch.Tensor(ypreds_test_balsa))

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
    
    # Baselines: Neo, Bao, Lero, Balsa
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


    labels = [
        'optimizer',
        'optimizer pruned',
        'Neo','Bao','Lero','Balsa',
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
    
    result_dict = {
        "q_ids":q_ids,
        "target":test_set.y.tolist(),
        "model base preds":ypreds_tens_test_org[:,0].tolist(), 
        "bao preds":bao_preds2_org_vanilla[:,0].tolist(), 
        "neo preds":neo_preds2_org_vanilla[:,0].tolist(), 
        "lero preds":ypreds_test_lero[:,0].tolist(), 
        "balsa preds":ypreds_test_balsa[:,0].tolist(),
        "mc dropout preds":ypreds_test_m_org.tolist(), 
        "optimizer cost":test_set.opt_cost.tolist(),
        "qerror":qerrors,
        "runtime_vals":rt_values,
        "subopt_vals":so_values,
        "labels":labels,
        "runtime details":runtime_details,
        "aggregate results":agg_res
        }

    return result_dict
    