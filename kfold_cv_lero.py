####################### kFolds Cross Validation ###########################
# TODO: the code should be able to run a subset of the models
import os
import time
import pickle
import json
import torch
import numpy as np
from pyg_data import queryPlanPGDataset_nosplit
from util.util import set_seed, load_model_params
from util.data_transform import *
from util.custom_loss import aleatoric_loss, rmse_loss, bce_loss
import pytorch_lightning as pl
from lcm.roq_model import lcm_pl as roq_model
from lcm.neo_bao_model import lcm_pl as neo_bao_model
from lcm.lero_model import LeroModelPairwise as leroPairwise
from lcm.balsa_model import balsa_simulation, balsa_model
from sklearn.model_selection import KFold,train_test_split
from single_test_lero import test
from util.eval_util import *
from torch_geometric.loader import DataLoader

models_path = os.path.join('.','lightning_models')
results_dir = os.path.join('.','results')

def kfold_cv(
    experiment_id = 'job',
    files_id = 'temp',labeled_data_dir = './labeled_data/',
    max_epochs = 1000, patience = 100,
    num_workers = 10, seed = 0, reload_data = False,
    target = 'latency', n_splits = 10,
    num_experiments = 5):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # sets seed for random, numpy, and torch cuda  
    set_seed(seed)

    torch.set_float32_matmul_precision('high')

    # Load train,v alidation, and test datasets
    print("loading full dataset")
    full_dataset = queryPlanPGDataset_nosplit(
        files_id = files_id,
        labeled_data_dir=labeled_data_dir,
        force_reload=reload_data,
        seed = seed,
        )
    print("{} queries and {} samples in training dataset: ".format(np.unique(np.array(full_dataset.query_id)).shape[0],full_dataset.len()))

    qids = np.array(full_dataset.query_id)
    unique_qids = np.unique(qids)

    kf = KFold(n_splits=n_splits, shuffle=True,random_state=seed)
    all_splits  = [k for k in kf.split(unique_qids)]
    
    qerror_dict = {}
    corr_dict = {}
    rt_res_dict = {}
    so_res_dict = {}
    agg_res_dict = {}
    
    for run_id in range(num_experiments):

        print('experiment:',experiment_id)
        print('run',run_id)
        set_seed(run_id+seed)

        q_ids = []
        targets = []
        base_model_preds = []
        bao_preds = []
        neo_preds=[]
        lero_preds=[]
        balsa_preds = []
        mc_dpout_preds = []
        opt_cost = []
        qerrors = []
        rt_values = []
        so_values = []
        labels = []

        for fold in range(n_splits):
            print("Dataset fold:",fold)
            train_val_idx,test_idx =all_splits[fold]
            # print("train_val_qids",train_val_qids)
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=.1,
                random_state=seed,
                shuffle=True
                )
            train_qids = unique_qids[train_idx]
            val_qids = unique_qids[val_idx]
            test_qids = unique_qids[test_idx]

            train_msk = np.isin(qids,train_qids)
            val_msk = np.isin(qids,val_qids)
            test_msk = np.isin(qids,test_qids)

            train_set = full_dataset[train_msk].copy()
            val_set = full_dataset[val_msk].copy()
            test_set = full_dataset[test_msk].copy()

            print("{} queries and {} samples in training dataset: ".format(np.unique(np.array(train_set.query_id)).shape[0],train_set.len()))
            print("{} queries and {} samples in validation dataset: ".format(np.unique(np.array(val_set.query_id)).shape[0],val_set.len()))
            print("{} queries and {} samples in test dataset: ".format(np.unique(np.array(test_set.query_id)).shape[0],test_set.len()))

            # Perform data transformations on inputs 
            # drop_const = dropConst(train_set)
            # train_set = drop_const(train_set)
            # val_set = drop_const(val_set)
            # test_set = drop_const(test_set)

            null_imp = nullImputation(train_set)
            train_set = null_imp(train_set)
            val_set = null_imp(val_set)
            test_set = null_imp(test_set)

            minmax_scale = minmaxScale(train_set)
            train_set = minmax_scale(train_set)
            val_set = minmax_scale(val_set)
            test_set = minmax_scale(test_set)

            # Perform data transformations on targets 
            yTransFunc = target_log_transform(train_set, target = target)

            train_set = yTransFunc.transform(train_set)
            val_set = yTransFunc.transform(val_set)
            test_set = yTransFunc.transform(test_set)

            plan_attr_shape = train_set[0].plan_attr.shape
            plan_ord_shape = train_set[0].plan_ord.shape
            graph_attr_shape= train_set[0].graph_attr.shape
            edge_attr_shape = train_set[0].edge_attr_s.shape
            node_attr_shape = train_set[0].x_s.shape

            # best_model_paths = []
            training_time=[]

            print('experiment:',experiment_id)            

            arch_labels = ['balsa','neo','bao','roq','lero']
            for idx,arch in enumerate(arch_labels):

                # load model hyper-parameters
                config = load_model_params(arch)

                # overwrite patience for lero as it does not need that many # TODO: handle this properly
                usedPatience = 10 if arch == 'lero' else patience

                # set the loss function
                if 'roq' in arch:
                    loss = aleatoric_loss(device=device)
                if 'lero' in arch:
                    loss = bce_loss()
                else:
                    loss = rmse_loss()

                if 'neo' in arch:
                    model = neo_bao_model(
                        num_node = node_attr_shape[0], 
                        node_dim = node_attr_shape[1],
                        edge_dim = edge_attr_shape[1],#fill_value =0, 
                        numPlanFeat=plan_attr_shape,
                        numPlanOrdFeat=plan_ord_shape,
                        numQueryGraphFeat = graph_attr_shape[0],
                        device = device, 
                        criterion = loss,
                        architecture = 'neo',
                        **config
                        )
                if 'bao' in arch:
                    model = neo_bao_model(
                        num_node = node_attr_shape[0], 
                        node_dim = node_attr_shape[1],
                        edge_dim = edge_attr_shape[1],#fill_value =0, 
                        numPlanFeat=plan_attr_shape,
                        numPlanOrdFeat=plan_ord_shape,
                        numQueryGraphFeat = graph_attr_shape[0],
                        device = device, 
                        criterion = loss,
                        architecture = 'bao',
                        **config
                        )
                
                if 'roq' in arch:
                    model = roq_model(
                        num_node = node_attr_shape[0], 
                        node_dim = node_attr_shape[1],
                        edge_dim = edge_attr_shape[1],#fill_value =0, 
                        numPlanFeat=plan_attr_shape,
                        numPlanOrdFeat=plan_ord_shape,
                        numQueryGraphFeat = graph_attr_shape[0],
                        with_var = True, device = device, 
                        criterion = loss,
                        **config
                        )
                
                if 'lero' in arch:
                    model = leroPairwise(
                        num_node = node_attr_shape[0], 
                        numPlanFeat=plan_attr_shape,
                        numPlanOrdFeat=plan_ord_shape,
                        numQueryGraphFeat = graph_attr_shape[0],
                        device = device, 
                        criterion = loss,
                        **config
                    )
                
                if 'balsa' in arch:
                    pretrain = config.pop('pretrain',False)
                    model = balsa_simulation(
                        num_node = node_attr_shape[0], 
                        node_dim = node_attr_shape[1],
                        edge_dim = edge_attr_shape[1],#fill_value =0, 
                        numPlanFeat=plan_attr_shape,
                        numPlanOrdFeat=plan_ord_shape,
                        numQueryGraphFeat = graph_attr_shape[0],
                        device = device, 
                        criterion = loss,
                        **config
                        )

                batch_size= config['batch_size']

                follow_batch = ['x_s']

                train_loader = DataLoader(
                    train_set, batch_size=batch_size,
                    persistent_workers=True,
                    shuffle=False, num_workers=num_workers, follow_batch=follow_batch
                    )
                val_loader = DataLoader(
                    val_set, batch_size=batch_size,
                    persistent_workers=True,
                    shuffle=False, num_workers=num_workers, follow_batch=follow_batch
                    )

                num_params = sum([np.prod(p.size()) for p in  model.parameters()])
                num_params = str(int(num_params/1000))
                num_q = str(round(np.unique(np.array(train_set.query_id)).shape[0]/1000))
                lr = str(model.lr)[:8]
                bs = str(model.batch_size)
                # do = str(model.dropout)[:5]
                model_name = f'{arch}_{experiment_id}_lr{lr}_bs{bs}_{num_q}kq_{num_params}kp_fold{fold}_run{run_id}'

                es = pl.callbacks.EarlyStopping(monitor='val_loss',patience=usedPatience, verbose=True)
                
                logger = pl.loggers.TensorBoardLogger('./lightning_logs', name = model_name)
                
                checkpointing = pl.callbacks.ModelCheckpoint(
                    dirpath=models_path,
                    monitor='val_loss',
                    save_top_k = 1,
                    filename=model_name+'{epoch}-{step}-{val_loss:.2f}',
                    verbose=True
                    )

                if 'balsa' in arch:
                    if pretrain == True:
                        pt_es = pl.callbacks.EarlyStopping(monitor='val_loss',patience=3, verbose=True)
                        pretrainer = pl.Trainer(
                            max_epochs=10,accelerator='gpu',
                            devices=1,
                            callbacks = [pt_es,checkpointing],
                            logger=logger,
                            log_every_n_steps=10
                            )
                        pretrainer.fit(model,train_loader,val_loader)
                        print('Pretraining done. loading best model...')
                        model = balsa_simulation.load_from_checkpoint(
                            checkpointing.best_model_path,
                            num_node = node_attr_shape[0], 
                            node_dim = node_attr_shape[1],
                            edge_dim = edge_attr_shape[1],#fill_value =0, 
                            numPlanFeat=plan_attr_shape,
                            numPlanOrdFeat=plan_ord_shape,
                            numQueryGraphFeat = graph_attr_shape[0],
                            device = device, 
                            criterion = loss,
                            **config
                            ).balsa_model
                        
                trainer = pl.Trainer(
                    max_epochs=max_epochs,accelerator='gpu',
                    devices=1, 
                    callbacks = [es, checkpointing], 
                    logger=logger, 
                    log_every_n_steps=10
                    )

                trainer.fit(model,train_loader,val_loader)

                print('Best {} model saved in \n{}'.format(model_name,checkpointing.best_model_path))

                # best_model_paths.append(checkpointing.best_model_path)
            
                if 'neo' in arch:
                    neo = neo_bao_model.load_from_checkpoint(
                        checkpointing.best_model_path,
                        num_node = node_attr_shape[0], 
                        node_dim = node_attr_shape[1],
                        edge_dim = edge_attr_shape[1],
                        numPlanFeat=plan_attr_shape,
                        numPlanOrdFeat=plan_ord_shape,
                        numQueryGraphFeat = graph_attr_shape[0],
                        device = device, 
                        criterion = loss,
                        architecture = 'neo',
                        **config
                        )

                if 'bao' in arch:
                    bao = neo_bao_model.load_from_checkpoint(
                        checkpointing.best_model_path,
                        num_node = node_attr_shape[0], 
                        node_dim = node_attr_shape[1],
                        edge_dim = edge_attr_shape[1],
                        numPlanFeat=plan_attr_shape,
                        numPlanOrdFeat=plan_ord_shape,
                        numQueryGraphFeat = graph_attr_shape[0],
                        device = device, 
                        criterion = loss,
                        architecture = 'bao',
                        **config
                        )
                    
                if 'roq' in arch:
                    roq = roq_model.load_from_checkpoint(
                        checkpointing.best_model_path, 
                        num_node = node_attr_shape[0], 
                        node_dim = node_attr_shape[1],
                        edge_dim = edge_attr_shape[1], 
                        numPlanFeat=plan_attr_shape,
                        numPlanOrdFeat=plan_ord_shape,
                        numQueryGraphFeat = graph_attr_shape[0],
                        with_var = True, device = device, 
                        criterion = loss,
                        **config
                        )
                
                if 'lero' in arch:
                    lero_pairwise = leroPairwise.load_from_checkpoint(
                        checkpointing.best_model_path,
                        num_node = node_attr_shape[0], 
                        numPlanFeat=plan_attr_shape,
                        numPlanOrdFeat=plan_ord_shape,
                        numQueryGraphFeat = graph_attr_shape[0],
                        device = device, 
                        criterion = loss,
                        **config
                        )
                    lero=lero_pairwise.lero_model
                
                if 'balsa' in arch:
                    balsa = balsa_model.load_from_checkpoint(
                        checkpointing.best_model_path,
                        num_node = node_attr_shape[0], 
                        node_dim = node_attr_shape[1],
                        edge_dim = edge_attr_shape[1],#fill_value =0, 
                        numPlanFeat=plan_attr_shape,
                        numPlanOrdFeat=plan_ord_shape,
                        numQueryGraphFeat = graph_attr_shape[0],
                        device = device, 
                        criterion = loss,
                        **config
                        )

                
                
            models = [balsa,neo,bao,roq,lero]

            # TODO: update single_test_lero.py
            result_dict = test(
                files_id = files_id, try_gpu_inf=True, 
                seed = seed, num_workers=5, show_fig = False,
                cons_ratio_sp = np.arange(0.0,3,0.1),
                prune_ratio_sp = np.arange(0,.75,0.05),
                models = models,
                val_set = val_set,
                test_set = test_set,
                yTransFunc = yTransFunc
                )
            
            q_ids.extend(result_dict["q_ids"])
            targets.extend(result_dict["target"])
            base_model_preds.extend(result_dict["model base preds"])
            bao_preds.extend(result_dict["bao preds"])
            neo_preds.extend(result_dict["neo preds"])
            lero_preds.extend(result_dict["lero preds"])
            balsa_preds.extend(result_dict["balsa preds"])
            mc_dpout_preds.extend(result_dict["mc dropout preds"])
            opt_cost.extend(result_dict["optimizer cost"])
            qerrors.append(result_dict["qerror"])
            rt_values.append(np.array(result_dict["runtime_vals"]))
            so_values.append(np.array(result_dict["subopt_vals"]))
            labels=result_dict["labels"]
        
        ##### convert all to arrays with desired dimensions ######
        q_ids=np.array(q_ids)
        targets=np.array(targets).squeeze()
        base_model_preds=np.array(base_model_preds).squeeze()
        bao_preds=np.array(bao_preds).squeeze()
        neo_preds=np.array(neo_preds).squeeze()
        lero_preds=np.array(lero_preds).squeeze()
        balsa_preds=np.array(balsa_preds).squeeze()
        mc_dpout_preds=np.array(mc_dpout_preds).squeeze()
        opt_cost=np.array(opt_cost).squeeze()
        qerrors=pd.concat(qerrors)
        rt_values = np.concatenate(rt_values,axis=1)
        so_values = np.concatenate(so_values,axis=1)

        # replace nans # TODO: find the root cause, handle this properly.
        lero_preds = np.nan_to_num(lero_preds, nan=0.0, posinf=1.0, neginf=0.0)
        balsa_preds = np.nan_to_num(balsa_preds, nan=0.0, posinf=1.0, neginf=0.0)

        ########### Compute Perason's correlations ############
        pearson_coef1, _ = compute_pearsonr(opt_cost,targets)
        pearson_coef4, _ = compute_pearsonr(base_model_preds,targets)
        pearson_coef5, _ = compute_pearsonr(bao_preds,targets)
        pearson_coef6, _ = compute_pearsonr(neo_preds,targets)
        pearson_coef7, _ = compute_pearsonr(mc_dpout_preds,targets)
        pearson_coef8, _ = compute_pearsonr(lero_preds,targets)
        pearson_coef9, _ = compute_pearsonr(balsa_preds,targets)

        ########### Compute Spearman's correlations ###########
        spearmanr_coef1, _ = compute_spearmanr(opt_cost,targets)
        spearmanr_coef4, _ = compute_spearmanr(base_model_preds,targets)
        spearmanr_coef5, _ = compute_spearmanr(bao_preds,targets)
        spearmanr_coef6, _ = compute_spearmanr(neo_preds,targets)
        spearmanr_coef7, _ = compute_spearmanr(mc_dpout_preds,targets)
        spearmanr_coef8, _ = compute_spearmanr(lero_preds,targets)
        spearmanr_coef9, _ = compute_spearmanr(balsa_preds,targets)

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
        
        #################### compare runtimes ####################
        runtime_details = pd.DataFrame(
            np.array(rt_values).transpose(),
            columns=labels,index=q_ids
            )
        
        runtime_res = compare_with_baseline(rt_values, labels).sort_values(by='total_runtime_change%')


        ################### analyze subopts #####################
        so_labels = [i for i in labels if i not in ['best']]

        subopt_details = pd.DataFrame(
            np.array(so_values).transpose(),
            columns=so_labels,index=q_ids
            )
        
        subopt_res = subopt_details.describe(
                        percentiles=[.5,.6,.7,.8,.90,.95,.99]).T
        subopt_res = subopt_res.loc[:,['mean','50%','60%','70%','80%','90%','95%','99%','max']].sort_values(by='mean')

        agg_res = runtime_res.merge(subopt_res,how='outer',left_index=True,right_index=True)


        qerror_dict[run_id]=qerrors
        corr_dict[run_id]=corr_df
        rt_res_dict[run_id]=runtime_details
        so_res_dict[run_id]=subopt_details
        agg_res_dict[run_id]=agg_res


    test_split = 'kfoldcv'
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

    return training_time

if __name__ == '__main__':

    # On Windows, call freeze_support() to support freezing the script into an executable
    # from multiprocessing import freeze_support
    # freeze_support()
    kfold_cv(
        experiment_id = 'job_main_5m03ex10kf_s1',
        files_id = 'job_enc_v2',
        labeled_data_dir = './labeled_data/',
        max_epochs = 1000,
        patience = 50,
        num_workers = 4,
        seed = 1,
        reload_data = True,
        target = 'latency',
        n_splits = 10,
        num_experiments = 1,
        )