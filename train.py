####################### TRAIN ROQ ###########################
# TODO: find a way to disable logging

import time
import pickle
import torch
import numpy as np
from pyg_data import queryPlanPGDataset
from util.util import set_seed
from util.data_transform import *
from util.custom_loss import aleatoric_loss, rmse_loss
import pytorch_lightning as pl
from lcm.roq_model import lcm_pl as roq
from lcm.neo_bao_model import lcm_pl as neo_bao


def experiment(experiment_id = 'job', architecture_p = 'bao',
         max_epochs = 1000, patience = 100, num_experiments = 5, num_workers = 10, seed = 0, reload_data = False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # sets seed for random, numpy, and torch cuda  
    set_seed(seed)

    roq_config ={
        'TCNNin': 64,
        'TCNNout': 32,
        'batch_size': 512,
        'dropout': 0.12333583157720673,
        'finalMLPin': 128,
        'finalMLPout': 128,
        'lr': 0.00035,
        'node_embd_dim': 16,
        'qp_attheads': 1,
        'qp_gnn_ls': 3,
        'query_module_out': 64
        }

    neo_config={
        'TCNNin': 512,
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
        'query_module_in': 128
        }

    bao_config={
        'TCNNin': 256,
        'TCNNout': 64,
        'batch_size': 512,
        'dropout': 0.15,
        'finalMLPin': 64,
        'finalMLPout': 32,
        'lr': 0.0005,
        'node_embd_dim': 16,
        'qp_attheads': 1,
        'qp_gnn_ls': 3,
        'query_module_out': 64
        }

    torch.set_float32_matmul_precision('high')

    # Load train,v alidation, and test datasets
    print("loading train")
    train_set = queryPlanPGDataset(split= 'train', files_id = 'job_main', force_reload=reload_data, val_samples = 0.1, test_samples = 0.1, seed = seed)
    print("Number of queries in training dataset: ",train_set.len())
    print("loading val")
    val_set = queryPlanPGDataset(split= 'val', files_id = 'job_main')
    print("Number of queries in vlidation dataset: ",val_set.len())
    print("loading test")
    test_set = queryPlanPGDataset(split= 'test', files_id = 'job_main',)
    print("Number of queries in test dataset: ",test_set.len())

    meanEdge, minEdge, maxEdge = getStats(train_set.edge_attr_s)
    uniqueCols = (minEdge == maxEdge)
    train_set._data.edge_attr_s = train_set.edge_attr_s[:,~uniqueCols]
    val_set._data.edge_attr_s = val_set.edge_attr_s[:,~uniqueCols]
    test_set._data.edge_attr_s = test_set.edge_attr_s[:,~uniqueCols]


    meanNode_s, minNode_s, maxNode_s = getStats(train_set.x_s)
    meanEdge, minEdge, maxEdge = getStats(train_set.edge_attr_s)

    nullImpFunc = NullImputation(meanNode=meanNode_s, meanEdge=meanEdge)
    train_set = nullImpFunc(train_set)
    val_set = nullImpFunc(val_set)
    test_set = nullImpFunc(test_set)

    inputTransFunc = inputTransform(nodeRange_s = (minNode_s,maxNode_s),
                                edgeRange= (minEdge,maxEdge))
    train_set = inputTransFunc(train_set)
    val_set = inputTransFunc(val_set)
    test_set = inputTransFunc(test_set)

    if 'roq' in architecture_p:
        yTransFunc = target_log_transform(train_set.y)
    else:
        yTransFunc = target_transform(train_set.y)

    train_set._data.y_t = yTransFunc.transform(train_set.y)
    val_set._data.y_t = yTransFunc.transform(val_set.y)
    test_set._data.y_t = yTransFunc.transform(test_set.y)

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

    train_loader = DataLoader(train_set, batch_size=batch_size,
                                persistent_workers=True,
                            shuffle=False, num_workers=num_workers, follow_batch=follow_batch)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            persistent_workers=True,
                            shuffle=False, num_workers=num_workers, follow_batch=follow_batch)

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
            model = neo_bao(num_node = node_attr_shape[0], 
                        node_dim = node_attr_shape[1],
                        edge_dim = edge_attr_shape[1],#fill_value =0, 
                        numPlanFeat=plan_attr_shape,
                        numPlanOrdFeat=plan_ord_shape,
                        numQueryGraphFeat = graph_attr_shape[0],
                        device = device, 
                        criterion = loss,
                        architecture = architecture_p,
                        **config)
        elif 'roq' in architecture_p:
            config = roq_config
            model = roq(num_node = node_attr_shape[0], 
                        node_dim = node_attr_shape[1],
                        edge_dim = edge_attr_shape[1],#fill_value =0, 
                        numPlanFeat=plan_attr_shape,
                        numPlanOrdFeat=plan_ord_shape,
                        numQueryGraphFeat = graph_attr_shape[0],
                        with_var = True, device = device, 
                        criterion = loss,
                        **config)

        # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        num_params = sum([np.prod(p.size()) for p in  model.parameters()])
        num_params = str(int(num_params/1000))
        num_q = str(round(np.unique(np.array(train_set.query_id)).shape[0]/1000))
        # model_name = 'lcm_full_gnn_v0'
        model_name = '{}_{}_lr{}_{}kq_{}kp_run{}'.format(
            architecture_p,experiment_id, 
            str(config['lr'])[:8],num_q,num_params,run_id)

        es = pl.callbacks.EarlyStopping(monitor='val_loss',patience=patience, verbose=True)
        
        logger = pl.loggers.TensorBoardLogger('./lightning_logs', name = model_name)
        
        checkpointing = pl.callbacks.ModelCheckpoint(
            dirpath='./lightning_models/',
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

    with open('./lightning_models/best_model_paths_{}_{}.pkl'.format(architecture_p,experiment_id), 'wb') as file:
        pickle.dump(best_model_paths, file)

    return training_time

if __name__ == '__main__':

    # On Windows, call freeze_support() to support freezing the script into an executable
    # from multiprocessing import freeze_support
    # freeze_support()

    experiment(
        experiment_id = 'temp', 
        architecture_p = 'bao',
        max_epochs = 100, 
        patience = 100, 
        num_experiments = 1, 
        num_workers = 10, 
        seed = 0,
        reload_data = False,
        )