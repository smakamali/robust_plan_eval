####################### TRAIN ###########################
# This will replace train.py in the future
# TODO: add support for training Balsa
 
import os
import time
import pickle
import torch
import numpy as np
from pyg_data import queryPlanPGDataset_withbenchmark
from util.util import set_seed, load_model_params
from util.data_transform import *
from util.custom_loss import aleatoric_loss, rmse_loss,bce_loss
import pytorch_lightning as pl
from lcm.roq_model import lcm_pl as roq
from lcm.neo_bao_model import lcm_pl as neo_bao
from lcm.lero_model import LeroModelPairwise as lero
from lcm.balsa_model import balsa_simulation as balsa
from torch_geometric.loader import DataLoader

models_path = os.path.join('.','lightning_models')

def train(
    experiment_id = 'job', architecture_p = 'roq', 
    files_id = 'temp', proc_files_id=None,benchmark_files_id='job_main', labeled_data_dir = './labeled_data/',
    max_epochs = 1000, patience = 100, num_experiments = 5, 
    num_workers = 10, seed = 0, reload_data = False, 
    num_samples = None,val_samples = 0.1,test_samples = 200,test_slow_samples=None,
    target = 'latency'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    # sets seed for random, numpy, and torch cuda  
    set_seed(seed)

    # load model hyper-parameters
    roq_config = load_model_params('roq')
    neo_config = load_model_params('neo')
    bao_config = load_model_params('bao')
    lero_config = load_model_params('lero')
    balsa_config = load_model_params('balsa')

    torch.set_float32_matmul_precision('high')

    # Load train, validation, and test datasets
    print("loading train")
    train_set = queryPlanPGDataset_withbenchmark(
        split= 'train', 
        files_id = files_id, 
        proc_files_id = proc_files_id,
        benchmark_files_id=benchmark_files_id,
        labeled_data_dir=labeled_data_dir,  
        force_reload=reload_data, 
        val_samples = val_samples, 
        test_samples = test_samples, 
        test_slow_samples=test_slow_samples, 
        seed = seed,
        num_samples = num_samples,
        exp_id=experiment_id
        )
    print("{} queries and {} samples in training dataset: ".format(np.unique(np.array(train_set.query_id)).shape[0],train_set.len()))

    print("loading val")
    val_set = queryPlanPGDataset_withbenchmark(
        split= 'val', 
        files_id = files_id,
        proc_files_id = proc_files_id,
        exp_id=experiment_id
        )
    print("{} queries and {} samples in vlidation dataset: ".format(np.unique(np.array(val_set.query_id)).shape[0],val_set.len()))
    
    # print("loading test")
    # test_set = queryPlanPairPGDataset(
    #     split= 'test', 
    #     files_id = files_id,
    #     exp_id=experiment_id
    #     )
    # print("{} queries and {} samples in test dataset: ".format(np.unique(np.array(test_set.query_id)).shape[0],test_set.len()))

    print("Data shapes:")
    for key in train_set.data.keys():
        if torch.is_tensor(train_set.data[key]):
            print(f"{key}: {train_set.data[key].shape}")


    plan_num_attr = int(train_set.data.plan_attr.shape[0]/len(train_set))
    plan_num_node = train_set.data.plan_attr.shape[1]
    node_costs = train_set.data.plan_attr.reshape(-1,plan_num_attr,plan_num_node).transpose(1,2).reshape(-1,plan_num_attr)[:,2]
    print("node_costs:", node_costs)

    # print("\nSlice information:")
    # for key in train_set.slices.keys():
    #     print(f"{key}: {train_set.slices[key]}")
    
#  -----------------------------------------------------------------
    # Perform data transformations on inputs 
    # drop_const = dropConst(train_set)
    # train_set = drop_const(train_set)
    # val_set = drop_const(val_set)
    # test_set = drop_const(test_set)

    null_imp = nullImputation(train_set)
    train_set = null_imp(train_set)
    val_set = null_imp(val_set)
    # test_set = null_imp(test_set)

    minmax_scale = minmaxScale(train_set)
    train_set = minmax_scale(train_set)
    val_set = minmax_scale(val_set)
    # test_set = minmax_scale(test_set)

    plan_num_attr = int(train_set.data.plan_attr.shape[0]/len(train_set))
    plan_num_node = train_set.data.plan_attr.shape[1]
    node_costs = train_set.data.plan_attr.reshape(-1,plan_num_attr,plan_num_node).transpose(1,2).reshape(-1,plan_num_attr)[:,2]
    print("node_costs:", node_costs)
    
    # raise SystemExit

    # Perform data transformations on targets 
    if 'lero' not in architecture_p:
        yTransFunc = target_log_transform(train_set, target = target)
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

    batch_size= 16 # roq_config['batch_size']


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
    if 'lero' in architecture_p:
        loss = bce_loss()
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
            
        elif 'lero' in architecture_p:
            config = lero_config
            model = lero(
                num_node = node_attr_shape[0], 
                numPlanFeat=plan_attr_shape,
                numPlanOrdFeat=plan_ord_shape,
                numQueryGraphFeat = graph_attr_shape[0],
                device = device, 
                criterion = loss,
                **config
            )

        elif 'balsa' in architecture_p:
            config = balsa_config
            pretrain = config.pop('pretrain',False)
            model = balsa(
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

        num_params = sum([np.prod(p.size()) for p in  model.parameters()])
        num_params = str(int(num_params/1000))
        num_q = str(round(np.unique(np.array(train_set.query_id)).shape[0]/1000))
        lr = str(config['lr'])[:8]
        bs = str(config['batch_size'])
        do = str(config['dropout'])[:5]
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

        if 'balsa' in architecture_p:
            if pretrain == True:
                pt_es = pl.callbacks.EarlyStopping(monitor='val_loss',patience=3, verbose=True)
                pretrainer = pl.Trainer(
                    max_epochs=10,accelerator='gpu',
                    devices=1,
                    callbacks = [pt_es,checkpointing],
                    logger=logger,
                    log_every_n_steps=1
                    )
                pretrainer.fit(model,train_loader,val_loader)
                print('Pretraining done. loading best model...')
                model = balsa.load_from_checkpoint(
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
            else:
                raise NotImplementedError("Balsa model must be pretrained")
                
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
    # -----------------------------------------------------------------

if __name__ == '__main__':

    # On Windows, call freeze_support() to support freezing the script into an executable
    # from multiprocessing import freeze_support
    # freeze_support()

    train(
        experiment_id = 'job_enc_v2_temp',
        architecture_p = 'balsa',
        files_id='job_enc_v2',
        proc_files_id='job_enc_v2_temp',
        benchmark_files_id ='job_enc_v2',
        labeled_data_dir='./labeled_data',
        max_epochs = 1000,
        patience = 20,
        num_experiments = 1,
        num_workers = 4,
        seed = 0,
        reload_data = True,
        num_samples = 100,
        val_samples = 0.1,
        test_samples = 0.1,
        test_slow_samples = 0.8,
        target = 'latency'
        )