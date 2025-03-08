# Robust Optimization of Queries (Roq)

This repository contains the code for a robust query optimization approach as presented in paper [Roq: Robust Query Optimization Based on a Risk-aware Learned Cost Model](https://arxiv.org/abs/2401.15210). This prototype is built based on `IBM Db2`. However, the concepts and ideas are directly transferable to any other relational database management system (RDBMS). This documentation covers:

* [Setting up a Db2 docker container with GPU support](#db2-docker-gpu)
* [Data Preparation](#data-prep)
* [Running Experiments](#experiments)


## <a id="db2-docker-gpu"></a> Setting up a Db2 docker container with GPU support

Istalling `NVIDIA Driver`, `CUDA Toolkit`, and `CUDA Container Toolkit` are required to make GPU of the host machine available to the containers. Doing so is pretty straight forward on a Linux host. Here we capture the steps for doing so on a Windows host, since the steps are a bit more involved:

1. Install [Docker Desktop and enable integration with WSL](https://docs.docker.com/desktop/wsl/#turn-on-docker-desktop-wsl-2).
2. Install [NVIDIA Driver for GPU Support on Windows](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#step-1-install-nvidia-driver-for-gpu-support).
3. Install `CUDA Toolkit` on WSL: 
    - [CUDA on WSL (nvidia.com)](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
    - [CUDA Toolkit Download for WSL](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network)
    ```
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-4
    ```
4. Install `CUDA Container Toolkit` on WSL:

    - [CUDA Container Toolkit for WSL](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
    ```
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
    sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

    sudo yum install -y nvidia-container-toolkit

    ```

5. Verify WSL has access to GPU(s):
    
    Run inside SWL to make sure it has access to GPU(s)
    
    ```
    nvidia-smi
    ```
    Test that a docker container can be created with access to GPU: 

    ```
    docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
    ```
6. build a Db2 container with access to GPU(s)
    ```
    docker run -itd --gpus all --name db2_container --privileged=true -p 50000:50000 -e LICENSE=accept -e DB2INST1_PASSWORD=db2inst1 -e DBNAME=test -v ./db2_db:/database ibmcom/db2
    ```

## <a id="data-prep"></a> Data Preparation
This repo assumes the dataset is already loaded in the database. Code for loading data in Db2 is available in the following repositories: 
- [IMDB](https://github.com/smakamali/setup_imdb_db2) 
- [TPCDS](https://github.com/smakamali/setup_tpcds_db2)

Steps involved in preparing data for a dataset and a workload:

1. **Installing Python dependencies:**

    ```
    pip install -r requirements.txt
    ```

2. **create a connection string file**
    ```
    cp conn_str_temp conn_str
    ```
    Modify the content of conn_str that follows this template:
    ```
    DATABASE=<schema_name>;HOSTNAME=localhost;PORT=50000;PROTOCOL=TCPIP;UID=db2inst1;PWD=*****;

    ```
3. **Create Explain tables**
    
    Start Db2 server:
    ```
    db2start
    ```
    Create explain tables which are needed for compiling queries:
    ```
    python -m scripts.create_exp_tables
    ```

4. **Collect database samples:**
    
    The first step involves collecting samples from each table of the database. These samples are used in the subsequent steps for capturing database statistics.

    ```
    python -m scripts.db_sampler
    ```

    Parameters to modify in `db_sampler.py`:

    - `schema_name` : schema name for which table samples will be collected, e.g.: `imdb`
    - `SAMPLE_SIZE` : max number of rows to be sampled from each table, e.g. `2000`


5. **Generate, encode, and label query-plan pairs:**

    This script includes several steps from compiling plans based on hintsets, collecting labels, encoding queries, and encoding plans. This can take several minutes depending on the number of queries provided in the `./input` direcotry.  
    ```
    python -m scripts.gen_label_plans
    ```
    
    Parameters to modify in `gen_label_plans.py`

    -  `max_num_queries` : Specify the max number of queries to process
    - `schema_name` : schema name used in the database
    - `encFileID` : a unique id for the generated dataset and any intermediary files in `internal_dir`
    - `conn_str_path` : path to a file containing a connection string to the database. a template is provided in `conn_str_temp`.
    - `input_dir` : the directory that contains input queries in 1 or multiple `*.sql` file(s), Default: `./input/`
    - `opt_plan_path` : the path used to store explain outputs and guidelines, Default: `./optimizer_plans/`
    - `internal_dir` : the path to store intermediary files, Default: `./internal/`
    - `sample_size` : number of samples used per table, set the same as the value used in `db_sampler.py`
    - `timeout_thr` : timeout threshold to avoid long running query/plans, Default: `60`
    - `dynamic_timeout` : determines whether dynamic timeout is used, Default: `False`
    - `dynamic_timeout_factor` : determines the multiplier for the dynamic timeout with respect to the optimizer's plan as a baseline, used only when `dynamic_timeout = True`, Default: `5`

6. **Using the PyG dataset:**

    After data generation, encoding, and labeling are complete, the dataset must be loaded in PyG dataset modules. The default PyG dataset module is available in `pyg_data.py`. This dataset module allows for loading `train`, `val`, and `test` splits as shown in the following example:
    ```
    from util.pyg_data import queryPlanPGDataset

    train_set = queryPlanPGDataset(
        split= 'train', 
        files_id = 'job_main',
        force_reload = False, 
        test_samples = 0.1
        val_samples = 0.1, 
        )
    val_set = queryPlanPGDataset(
        split= 'val', 
        files_id = 'job_main'
        )
    test_set = queryPlanPGDataset(
        split= 'test', 
        files_id = 'job_main'
        )
    ```

    The available parameters:
    - `split` : can be either `train`, `val`, or `test`
    - `files_id` : the unique ID of the generated data. This must match `encFileID` used in the previous step. 
    - `force_reload` : By default PyG Dataset module stores the datasets to disk after processing. In the subsequent calls, the processed dataset is loaded from disk, unless force `force_reload = True`, in which case the data is processed and saved to disk again.
        - Note: `force_reload` must not be set to `True` for `split = 'val'` or  `'test'`. Otherwise, the different splits might end up having overlaps.
    - `test_samples` : The ratio of the dataset used for testing.
    - `val_samples` : The ratio of the remaining data used for validation. 
    
    Other parameters:
    - `internal_dir` : The path to the internal files. Default: `./internal/`
    - `num_samples` : If specified will be used to subsample the entire dataset before splitting. Default: `False`
    - `seed`: A random seed to control reproducibility of dataset splitting. 

## <a id="experiments"></a>Running Experiments - Train a model

To train the model use `train.py` script with the desired parameters. An example use case is provided below.

### Example Execution

```python
if __name__ == '__main__':
    train(
        experiment_id = 'job_syn_slow',         # Unique experiment identifier
        architecture_p = 'roq',                 # Model architecture to use ('roq')
        files_id = 'job_syn_all',               # File ID for the dataset
        benchmark_files_id = 'job_main',        # File ID for benchmark dataset
        labeled_data_dir = './labeled_data',    # Directory containing labeled data
        max_epochs = 50,                        # Maximum number of epochs to train
        patience = 100,                         # Early stopping patience 
        num_experiments = 1,                    # Number of training runs (experiments)
        num_workers = 3,                        # Number of workers for data loading
        seed = 0,                               # Seed for reproducibility
        reload_data = True,                     # Reload data flag
        val_samples = 500,                      # Number of validation samples
        test_samples = 500,                     # Number of test samples
        test_slow_samples = 0.8,                # Proportion of slow queries in test set
        target = 'latency'                      # Prediction target (latency) - alternatively it can be set to `cost`
    )
```

### Parameters

- `experiment_id`: Unique identifier for the experiment.
- `architecture_p`: The model architecture to use (`roq`, `neo`, `bao`).
- `files_id`: The ID of the training dataset file.
- `benchmark_files_id`: The ID of the benchmark dataset file.
- `labeled_data_dir`: Directory where labeled data files are stored.
- `max_epochs`: Maximum number of epochs to train the model.
- `patience`: Number of epochs to wait for improvement before early stopping. A high number ensures training until convergence. A learning rate scheduler is used to avoid overfitting
- `num_experiments`: Number of independent training runs to execute.
- `num_workers`: Number of workers for data loading.
- `seed`: Random seed for reproducibility.
- `reload_data`: Whether to reload data.
- `val_samples`: Number of validation samples.
- `test_samples`: Number of test samples.
- `test_slow_samples`: Proportion of slow queries in the test set.
- `target`: Target variable for prediction (`latency` in this case, alternatively it can be set to `cost`).

### Model Architectures

This script supports multiple ML-based learned cost models for query plan latency prediction:
- **ROQ** is a GNN-based architecture specifically designed for query plan latency prediction. It offers a holistic framework based on a risk-aware learning approach. Roq includes a novel formalization of the notion of robustness in the context of query optimization and a principled approach for its quantification and measurement based on approximate probabilistic ML. It also includes novel strategies and algorithms for query plan evaluation and selection. Roq includes a novel learned cost model that is designed to predict the cost of query execution and the associated risks and performs query optimization accordingly [1].

- **NEO** is a learned query optimizer that uses a learned cost model that takes query and plan attributes as input. It feeds the query attributes to a stack of MLP layers and appends the resulting vector to the attributes of every node in the vectorized plan tree. It then feeds the extended plan trees to a stack of TCNN layers followed by another stack of MLPs which gradually reduce the dimensionality of the vector and ultimately predict the execution time [2].
- **BAO** uses a more lightweight cost model that takes plans as input and predicts latency. At each node of the vectorized plan tree, in addition to node type, it captures the estimated cardinality and cost [3].
- **Lero** is a more recent work that proposes a learning-to-rank approach to a learned query optimizer. It learns to compare pairs of plans in the search space and create a total order. The model is trained using a pairwise ranking loss function. Additionally, it enumerates pairs of subplans traversed during search and uses them as additional samples to train the model [4].
- **Balsa** is a learned query optimizer that avoids relying on expert demonstrations. It bootstraps from a simple simulator and then fine-tunes its deep reinforcement learning model using real query execution latency feedback [5].

### Data Preprocessing

The following preprocessing steps are applied to the dataset:
1. Drop constant features.
2. Impute missing values.
3. Scale the data using min-max normalization.
4. Apply a logarithmic transformation to the target variable (`latency`).

### Training

The script trains the model using PyTorch Lightning, applying early stopping based on validation loss. Checkpoints are saved for the best-performing models.

### Logging and Checkpoints

Training logs are stored using TensorBoard, and model checkpoints are saved to the `./lightning_models` directory. You can monitor training progress in real-time by running:

```bash
tensorboard --logdir=./lightning_logs
```

### Example Execution

The following command will start the training process given the parameters specified in `scripts.train`:

```bash
python -m scripts.train
```

Best models will be saved in the `./lightning_models` directory, and logs can be found in `./lightning_logs`.

## Running Multiple Experiments

USe the script `run_experiments.py` This script orchestrates the training of multiple Graph Neural Network (GNN) architectures for the task of query latency prediction. The model architectures supported are:

- **Neo**
- **Bao**
- **Roq**
- **Lero**
- **Balsa**

It calls the `train()` function from the `train.py` module, training each architecture for multiple experiments and saving the training times to a results file.

### Example Usage

First set the parameters in `run_experiments.py`:

```python
# specify the model to be trained
architectures = [
    'bao',
    'neo',
    'roq',
    'lero',
    'balsa',
]
experiment_id = 'all_models_ce'

training_time_dict  = {}
for arch in architectures:
    training_time = train(
        experiment_id = experiment_id,
        architecture_p = arch,
        files_id= 'ceb_1000',
        benchmark_files_id = 'job_main',
        labeled_data_dir='./labeled_data',
        max_epochs = 1000,
        patience = 50,
        num_experiments = 5,
        num_workers = 5,
        seed = 3,
        reload_data = False,
        val_samples = 0.1,
        test_samples = 100,
        test_slow_samples = None,
        target = 'latency'
    )
    training_time_dict[arch] = training_time
```

To run the script:

```bash
python -m scripts.run_experiments
```
### Output

After running the script, the training times for each architecture will be saved in a pickle file in the `results` directory:

```
results/training_time_<experiment_id>.pkl
```
## Model Evaluation
For model evaluation use `eval.py` module. This module performs the following tasks:

1. Loads the trained models from checkpoints
2. Makes inferences
3. Tunes the plan selection strategy hyperparameters using the validation set
4. Evaluates the models based on the test set, which involves: computing the q-error, correlations, runtime, suboptimality, and aggregated results and saves them in `qerror_dict{<experiment_id>}_{<test_split>}.pkl`,`corr_dict{<experiment_id>}_{<test_split>}.pkl`,`rt_res_dict{<experiment_id>}_{<test_split>}.pkl`,`so_res_dict{<experiment_id>}_{<test_split>}.pkl`, and `agg_res_dict{<experiment_id>}_{<test_split>}.pkl`respecively.

### Usage example
```bash
python -m scripts.eval
```

## References:
1. [Roq: Robust Query Optimization Based on a Risk-aware Learned Cost Model](https://arxiv.org/abs/2401.15210)
2. [Neo: A Learned Query Optimizer](https://www.vldb.org/pvldb/vol12/p1705-marcus.pdf)
3. [Bao: Making Learned Query Optimization Practical](dl.acm.org/doi/pdf/10.1145/3448016.3452838)
4. [Lero: A Learning-to-Rank Query Optimizer](https://dlnext.acm.org/doi/abs/10.14778/3583140.3583160)
5. [Balsa: Learning a Query Optimizer Without Expert Demonstrations](https://zongheng.me/pubs/balsa-sigmod2022.pdf)