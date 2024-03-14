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
    python create_exp_tables.py
    ```

4. **Collect database samples:**
    
    The first step involves collecting samples from each table of the database. These samples are used in the subsequent steps for capturing database statistics.

    ```
    python db_sampler.py
    ```

    Parameters to modify in `db_sampler.py`:

    - `schema_name` : schema name for which table samples will be collected, e.g.: `imdb`
    - `SAMPLE_SIZE` : max number of rows to be sampled from each table, e.g. `2000`


5. **Generate, encode, and label query-plan pairs:**

    This script includes several steps from compiling plans based on hintsets, collecting labels, encoding queries, and encoding plans. This can take several minutes depending on the number of queries provided in the `./input` direcotry.  
    ```
    python gen_label_plans.py
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
    from pyg_data import queryPlanPGDataset

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

## <a id="experiments"></a>Running Experiments

Coming soon ...