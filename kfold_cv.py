####################### kFolds Cross Validation ###########################

import os
import time
import pickle
import json
import torch
import numpy as np
from pyg_data import queryPlanPGDataset_nosplit
from util.util import set_seed, load_model_params
from util.data_transform import *
from util.custom_loss import aleatoric_loss, rmse_loss
import pytorch_lightning as pl
from lcm.roq_model import lcm_pl as roq_model
from lcm.neo_bao_model import lcm_pl as neo_bao_model
from sklearn.model_selection import KFold,train_test_split
from single_test import test

models_path = os.path.join('.','lightning_models')
results_dir = os.path.join('.','results')

def kfold_cv(
    experiment_id = 'job', 
    files_id = 'temp',labeled_data_dir = './labeled_data/',
    max_epochs = 1000, patience = 100, 
    num_workers = 10, seed = 0, reload_data = False, 
    target = 'latency', n_splits = 10):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # sets seed for random, numpy, and torch cuda  
    set_seed(seed)

    # load model hyper-parameters
    roq_config = load_model_params('roq')
    neo_config = load_model_params('neo')
    bao_config = load_model_params('bao')

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
    for fold in range(n_splits):

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
        yTransFunc = target_log_transform(train_set, target = target)

        # if 'roq' in architecture_p:
        #     yTransFunc = target_log_transform(train_set, target = target)
        # else:
        #     yTransFunc = target_transform(train_set, target = target)

        train_set = yTransFunc.transform(train_set)
        val_set = yTransFunc.transform(val_set)
        test_set = yTransFunc.transform(test_set)

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

        from torch_geometric.loader import DataLoader

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

        # if 'roq' in architecture_p:
        aleatoricloss = aleatoric_loss(device=device)
        # else:
        rmseloss = rmse_loss()

        best_model_paths = []
        training_time=[]

        # for run_id in range(num_experiments):
        print('experiment:',experiment_id)
        # print('architecture:',architecture_p)
        # print('run',run_id)
        # set_seed(run_id+seed)

        tic = time.time()

        # if 'neo' in architecture_p or 'bao' in architecture_p:
        #     if 'neo' in architecture_p:
        #         config = neo_config
        #     else:
        #         config = bao_config
        neo = neo_bao_model(
            num_node = node_attr_shape[0], 
            node_dim = node_attr_shape[1],
            edge_dim = edge_attr_shape[1],#fill_value =0, 
            numPlanFeat=plan_attr_shape,
            numPlanOrdFeat=plan_ord_shape,
            numQueryGraphFeat = graph_attr_shape[0],
            device = device, 
            criterion = rmseloss,
            architecture = 'neo',
            **neo_config
            )
        
        bao = neo_bao_model(
            num_node = node_attr_shape[0], 
            node_dim = node_attr_shape[1],
            edge_dim = edge_attr_shape[1],#fill_value =0, 
            numPlanFeat=plan_attr_shape,
            numPlanOrdFeat=plan_ord_shape,
            numQueryGraphFeat = graph_attr_shape[0],
            device = device, 
            criterion = rmseloss,
            architecture = 'bao',
            **bao_config
            )
        
        # elif 'roq' in architecture_p:
        #     config = roq_config
        roq = roq_model(
            num_node = node_attr_shape[0], 
            node_dim = node_attr_shape[1],
            edge_dim = edge_attr_shape[1],#fill_value =0, 
            numPlanFeat=plan_attr_shape,
            numPlanOrdFeat=plan_ord_shape,
            numQueryGraphFeat = graph_attr_shape[0],
            with_var = True, device = device, 
            criterion = aleatoricloss,
            **roq_config
            )

        arch_labels = ['neo','bao','roq']
        for idx,model in enumerate([neo,bao,roq]):

            num_params = sum([np.prod(p.size()) for p in  model.parameters()])
            num_params = str(int(num_params/1000))
            num_q = str(round(np.unique(np.array(train_set.query_id)).shape[0]/1000))
            lr = str(model.lr)[:8]
            bs = str(model.batch_size)
            do = str(model.dropout)[:5]
            model_name = '{}_{}_lr{}_bs{}_do{}_{}kq_{}kp_fold{}'.format(
                arch_labels[idx],experiment_id, 
                lr,bs,do,num_q,num_params,fold)

            es = pl.callbacks.EarlyStopping(monitor='val_loss',patience=patience, verbose=True)
            
            logger = pl.loggers.TensorBoardLogger('./lightning_logs', name = model_name)
            
            checkpointing = pl.callbacks.ModelCheckpoint(
                dirpath=models_path,
                monitor='val_loss',
                save_top_k = 1,
                filename=model_name+'{epoch}-{step}-{val_loss:.2f}',
                verbose=True
                )

            trainer = pl.Trainer(
                max_epochs=max_epochs,accelerator='gpu',
                devices=1, 
                callbacks = [es, checkpointing], 
                logger=logger, 
                log_every_n_steps=10
                )

            trainer.fit(model,train_loader,val_loader)

            print('Best {} model saved in \n{}'.format(model_name,checkpointing.best_model_path))

            best_model_paths.append(checkpointing.best_model_path)
          
            if idx == 0:
                neo = neo_bao_model.load_from_checkpoint(
                    checkpointing.best_model_path,
                    num_node = node_attr_shape[0], 
                    node_dim = node_attr_shape[1],
                    edge_dim = edge_attr_shape[1],
                    numPlanFeat=plan_attr_shape,
                    numPlanOrdFeat=plan_ord_shape,
                    numQueryGraphFeat = graph_attr_shape[0],
                    device = device, 
                    criterion = rmseloss,
                    architecture = 'neo',
                    **neo_config
                    )

            elif idx == 1:
                bao = neo_bao_model.load_from_checkpoint(
                    checkpointing.best_model_path,
                    num_node = node_attr_shape[0], 
                    node_dim = node_attr_shape[1],
                    edge_dim = edge_attr_shape[1],
                    numPlanFeat=plan_attr_shape,
                    numPlanOrdFeat=plan_ord_shape,
                    numQueryGraphFeat = graph_attr_shape[0],
                    device = device, 
                    criterion = rmseloss,
                    architecture = 'bao',
                    **bao_config
                    )
                
            elif idx ==2:
                roq = roq_model.load_from_checkpoint(
                    checkpointing.best_model_path, 
                    num_node = node_attr_shape[0], 
                    node_dim = node_attr_shape[1],
                    edge_dim = edge_attr_shape[1], 
                    numPlanFeat=plan_attr_shape,
                    numPlanOrdFeat=plan_ord_shape,
                    numQueryGraphFeat = graph_attr_shape[0],
                    with_var = True, device = device, 
                    criterion = aleatoricloss,
                    **roq_config
                    )
            
        models = [neo,bao,roq]

        qerrors, corr_df, runtime_details, subopt_details, agg_res = test(
            files_id = files_id, try_gpu_inf=True, 
            seed = seed, num_workers=5, show_fig = False,
            cons_ratio_sp = np.arange(0.0,3,0.1),
            prune_ratio_sp = np.arange(0,.75,0.05),
            models = models,
            val_set = val_set,
            test_set = test_set,
            yTransFunc = yTransFunc
            )

        qerror_dict[fold]=qerrors
        corr_dict[fold]=corr_df
        rt_res_dict[fold]=runtime_details
        so_res_dict[fold]=subopt_details
        agg_res_dict[fold]=agg_res

        toc = time.time()
        training_time.append(toc-tic)

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

        # with open(os.path.join(models_path,'best_model_paths_{}_{}.pkl'.format(archs[idx],experiment_id)), 'wb') as file:
        #     pickle.dump(best_model_paths, file)

    return training_time

if __name__ == '__main__':

    # On Windows, call freeze_support() to support freezing the script into an executable
    # from multiprocessing import freeze_support
    # freeze_support()

    kfold_cv(
        experiment_id = 'job_main', 
        files_id = 'job_main',
        labeled_data_dir = './labeled_data/',
        max_epochs = 1000,
        patience = 100,
        num_workers = 10,
        seed = 0,
        reload_data = False, 
        target = 'latency',
        n_splits = 10
        )