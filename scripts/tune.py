####################### TUNE ROQ #######################

import os
import pytorch_lightning as pl
import ray
import numpy as np
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from torch_geometric.loader import DataLoader
from util.pyg_data import queryPlanPGDataset_withbenchmark
from util.util import set_seed
from util.data_transform import *
from lcm.roq_model import lcm_pl as roq
from lcm.neo_bao_model import lcm_pl as neo_bao
from util.custom_loss import aleatoric_loss, rmse_loss

# Experiment parameters
architecture_p = 'roq'
experiment_id = 'job_r41'
total_cpus = os.cpu_count()
cpus_per_trial = 2 # determines the number of cpus used by each experiment. when OOM happens, reducing this value may help to resolve by reducing the number of concurrent experiments
gpus_per_trial = cpus_per_trial/total_cpus*0.95 # determines the number (or ratio) of gpus used by each experiment
num_experiments=150
num_epochs=150
patience=25
num_samples = None

# queryPlanPGDataset data module parameteres
files_id = 'job_v2.2'
proc_files_id = 'job_v2.2'
benchmark_files_id = 'job_v2.2'
labeled_data_dir = './labeled_data/job/'
results_dir ='./param_data/'
seed = 110
reload_data = True
val_samples = 0.1
test_samples = 0.01
test_slow_samples = 0.0

if not os.path.exists(results_dir):
    os.mkdir(results_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

set_seed(seed)

# Load train,v alidation, and test datasets
print("loading train")
train_set = queryPlanPGDataset_withbenchmark(
    split= 'train', 
    files_id = files_id,
    proc_files_id=proc_files_id,
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
    proc_files_id=proc_files_id,
    exp_id=experiment_id
    )
print("{} queries and {} samples in vlidation dataset: ".format(np.unique(np.array(val_set.query_id)).shape[0],val_set.len()))
    
# print("loading test")
# test_set = queryPlanPGDataset_withbenchmark(
#     split= test_split, 
#     files_id = files_id,
#     exp_id=experiment_id
#     )
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
yTransFunc = target_log_transform(train_set)
# if 'roq' in architecture_p:
#     yTransFunc = target_log_transform(train_set)
# else:
#     yTransFunc = target_transform(train_set)

train_set = yTransFunc.transform(train_set)
val_set = yTransFunc.transform(val_set)

# config = {
#     # Roq's param search space
#     'node_embd_dim': tune.choice([16,32,64]),
#     'query_module_out': tune.choice([256,128,64]),
#     'qp_attheads': tune.randint(1,3),
#     'qp_gnn_ls':tune.randint(2,5),
#     'query_graph_embd_dim': tune.choice([32,64,128]),

#     'TCNNin':tune.choice([512]),
#     'TCNNout':tune.choice([128]),

#     'finalMLPin': tune.choice([128]),
#     'finalMLPout': tune.choice([32]),

#     'batch_size': tune.choice([128]),
#     # 'dropout': tune.uniform(0.05,.2),
#     'dropout': tune.choice([.1]),
#     # 'lr': tune.qloguniform(1e-4, 1e-2, 5e-5),
#     'lr': tune.choice([1e-3]),
#     "rlrop_patience": tune.choice([10]),
#     # "rlrop_factor": tune.uniform(0.9,0.1)
#     "rlrop_factor": tune.choice([0.1])
#     }

config = {
#     # Roq's param search space for fine tuning lr
    'node_embd_dim': tune.choice([32]),
    'qp_attheads': tune.choice([1]),
    'qp_gnn_ls':tune.choice([4]),
    'query_graph_embd_dim': tune.choice([64]),
    'query_module_out': tune.choice([128]),

    'TCNNin':tune.choice([256]),
    'TCNNout':tune.choice([64]),

    'finalMLPin': tune.choice([256]),
    'finalMLPout': tune.choice([64]),
    
    'batch_size': tune.choice([32,64,128]),
    'dropout': tune.uniform(0.05,.2),
    'lr': tune.qloguniform(1e-4, 1e-2, 5e-5),
    "rlrop_patience": tune.choice([5,6,7,8,9,10]),
    "rlrop_factor": tune.uniform(0.9,0.1)
}

plan_attr_shape = train_set[0].plan_attr.shape
plan_ord_shape = train_set[0].plan_ord.shape
graph_attr_shape= train_set[0].graph_attr.shape
edge_attr_shape = train_set[0].edge_attr_s.shape
node_attr_shape = train_set[0].x_s.shape


def train_model(config,num_epochs,data,cpus,architecture_p):
    batch_size = config['batch_size']
    follow_batch = ['x_s']
    print('batch_size', batch_size)
    train_ref,val_ref = data
    train_loader = DataLoader(
        ray.get(train_ref), batch_size=batch_size,
        persistent_workers=True,
        shuffle=False, num_workers=cpus, 
        follow_batch=follow_batch
        )
    val_loader = DataLoader(
        ray.get(val_ref), batch_size=batch_size,
        persistent_workers=True,
        shuffle=False, num_workers=cpus, 
        follow_batch=follow_batch
        )
    
    torch.set_float32_matmul_precision('medium')

    if 'roq' in architecture_p:
        
        criterion = aleatoric_loss(device=device)

        model = roq(
            num_node = node_attr_shape[0], 
            node_dim = node_attr_shape[1],
            edge_dim = edge_attr_shape[1],
            numPlanFeat=plan_attr_shape,
            numPlanOrdFeat=plan_ord_shape,
            numQueryGraphFeat = graph_attr_shape[0],
            with_var = True, device = device,
            criterion = criterion,
            **config
            )
    
    elif 'neo' in architecture_p or 'bao' in architecture_p:
        
        criterion = rmse_loss()

        model = neo_bao(
            num_node = node_attr_shape[0], 
            node_dim = node_attr_shape[1],
            edge_dim = edge_attr_shape[1],
            numPlanFeat=plan_attr_shape,
            numPlanOrdFeat=plan_ord_shape,
            numQueryGraphFeat = graph_attr_shape[0],
            device = device, 
            criterion = criterion,
            architecture = architecture_p,
            **config
            )

    
    metrics = {"loss": "val_loss"}
    ray_cb = TuneReportCheckpointCallback(
        metrics, save_checkpoints = False ,on="validation_end"
        )
    
    es = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience, 
        verbose=False
        )
    
    trainer = pl.Trainer(
        max_epochs=num_epochs,accelerator='auto',
        devices=1, callbacks = [ray_cb,es],
        log_every_n_steps=1,
        enable_progress_bar=False
        )
    
    trainer.fit(model,train_loader,val_loader)

def tune_lcm_gnn_pl(num_experiments=1000, num_epochs=400, cpus = 2, gpus_per_trial = 0, architecture_p = 'roq'):
    
    train_ref = ray.put(train_set)
    val_ref = ray.put(val_set)    
    data = (train_ref,val_ref)
    
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=2,
        reduction_factor=2
        )

    train_fn_with_parameters = tune.with_parameters(
        train_model,
        num_epochs=num_epochs,
        data=data,
        cpus=cpus,
        architecture_p=architecture_p,
        # num_gpus=gpus_per_trial,
        # query_processor = 'gnn'
        # static_config=static_config,
        # batch_size=batch_size
        # data_dir=data_dir
        )
    resources_per_trial = {"cpu": cpus, "gpu": gpus_per_trial}
    
    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial
            ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_experiments,
            ),
        run_config=air.RunConfig(
            name="tune_lcm_{}".format(architecture_p),
#             progress_reporter=reporter,
            ),
        param_space=config
        )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)
    return results

results = tune_lcm_gnn_pl(num_experiments=num_experiments, num_epochs=num_epochs, cpus = cpus_per_trial, gpus_per_trial = gpus_per_trial, architecture_p = architecture_p)

# save param tuning results to disk
results_df = results.get_dataframe()
results_df.to_csv('./param_data/{}_{}_param_tun_res.csv'.format(architecture_p,experiment_id))
