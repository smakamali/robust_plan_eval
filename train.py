####################### TRAIN ROQ ###########################
# TODO: find a way to disable tensorboard logging

import os
import time
import pickle
import json
import torch
import numpy as np
from pyg_data import queryPlanPGDataset,queryPlanPGDataset_withbenchmark
from util.util import set_seed, load_model_params
from util.data_transform import *
from util.custom_loss import aleatoric_loss, rmse_loss
import pytorch_lightning as pl
from lcm.roq_model import lcm_pl as roq
from lcm.neo_bao_model import lcm_pl as neo_bao

models_path = os.path.join('.','lightning_models')

def train(
    experiment_id = 'job', architecture_p = 'roq', 
    files_id = 'temp',benchmark_files_id='job_main', labeled_data_dir = './labeled_data/',
    max_epochs = 1000, patience = 100, num_experiments = 5, 
    num_workers = 10, seed = 0, reload_data = False, 
    val_samples = 0.1,test_samples = 200,test_slow_samples=None,
    target = 'latency'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # sets seed for random, numpy, and torch cuda  
    set_seed(seed)

    # load model hyper-parameters
    roq_config = load_model_params('roq')
    neo_config = load_model_params('neo')
    bao_config = load_model_params('bao')

    torch.set_float32_matmul_precision('high')

    # Load train,v alidation, and test datasets
    print("loading train")
    train_set = queryPlanPGDataset_withbenchmark(
        split= 'train', 
        files_id = files_id, 
        benchmark_files_id=benchmark_files_id,
        labeled_data_dir=labeled_data_dir,  
        force_reload=reload_data, 
        val_samples = val_samples, 
        test_samples = test_samples, 
        test_slow_samples=test_slow_samples, 
        seed = seed,
        exp_id=experiment_id
        )
    print("{} queries and {} samples in training dataset: ".format(np.unique(np.array(train_set.query_id)).shape[0],train_set.len()))

    print("loading val")
    val_set = queryPlanPGDataset_withbenchmark(
        split= 'val', 
        files_id = files_id,
        exp_id=experiment_id
        )
    print("{} queries and {} samples in vlidation dataset: ".format(np.unique(np.array(val_set.query_id)).shape[0],val_set.len()))
    
    # print("loading test")
    # test_set = queryPlanPGDataset_withbenchmark(split= 'test', files_id = files_id)
    # print("{} queries and {} samples in test dataset: ".format(np.unique(np.array(test_set.query_id)).shape[0],test_set.len()))


    # Perform data transformations on inputs 
    drop_const = dropConst(train_set)
    train_set = drop_const(train_set)
    val_set = drop_const(val_set)
    # test_set = drop_const(test_set)

    null_imp = nullImputation(train_set)
    train_set = null_imp(train_set)
    val_set = null_imp(val_set)
    # test_set = null_imp(test_set)

    minmax_scale = minmaxScale(train_set)
    train_set = minmax_scale(train_set)
    val_set = minmax_scale(val_set)
    # test_set = minmax_scale(test_set)

    # Perform data transformations on targets 
    yTransFunc = target_log_transform(train_set, target = target)

    # if 'roq' in architecture_p:
    #     yTransFunc = target_log_transform(train_set, target = target)
    # else:
    #     yTransFunc = target_transform(train_set, target = target)

    train_set = yTransFunc.transform(train_set)
    val_set = yTransFunc.transform(val_set)
    # test_set = yTransFunc.transform(test_set)

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

    if 'roq' in architecture_p:
        loss = aleatoric_loss(device=device)
    else:
        loss = rmse_loss()

    best_model_paths = []
    training_time=[]

    for run_id in range(num_experiments):
        print('experiment:',experiment_id)
        print('architecture:',architecture_p)
        print('run',run_id)
        set_seed(run_id+seed)

        tic = time.time()

        if 'neo' in architecture_p or 'bao' in architecture_p:
            if 'neo' in architecture_p:
                config = neo_config
            else:
                config = bao_config
            model = neo_bao(
                num_node = node_attr_shape[0], 
                node_dim = node_attr_shape[1],
                edge_dim = edge_attr_shape[1],#fill_value =0, 
                numPlanFeat=plan_attr_shape,
                numPlanOrdFeat=plan_ord_shape,
                numQueryGraphFeat = graph_attr_shape[0],
                device = device, 
                criterion = loss,
                architecture = architecture_p,
                **config
                )
        elif 'roq' in architecture_p:
            config = roq_config
            model = roq(
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

        # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        num_params = sum([np.prod(p.size()) for p in  model.parameters()])
        num_params = str(int(num_params/1000))
        num_q = str(round(np.unique(np.array(train_set.query_id)).shape[0]/1000))
        lr = str(config['lr'])[:8]
        bs = str(config['batch_size'])
        do = str(config['dropout'])[:5]
        # model_name = 'lcm_full_gnn_v0'
        model_name = '{}_{}_lr{}_bs{}_do{}_{}kq_{}kp_run{}'.format(
            architecture_p,experiment_id, 
            lr,bs,do,num_q,num_params,run_id)

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

        toc = time.time()
        training_time.append(toc-tic)

    with open(os.path.join(models_path,'best_model_paths_{}_{}.pkl'.format(architecture_p,experiment_id)), 'wb') as file:
        pickle.dump(best_model_paths, file)

    return training_time

if __name__ == '__main__':

    # On Windows, call freeze_support() to support freezing the script into an executable
    # from multiprocessing import freeze_support
    # freeze_support()

    train(
        experiment_id = 'job_syn_slow',
        architecture_p = 'roq',
        files_id='job_syn_all',
        benchmark_files_id ='job_main',
        labeled_data_dir='./labeled_data',
        max_epochs = 50,
        patience = 100,
        num_experiments = 1,
        num_workers = 3,
        seed = 0,
        reload_data = True,
        val_samples = 500,
        test_samples = 500,
        test_slow_samples = 0.8,
        target = 'latency'
        )