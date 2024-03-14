# Robust Optimization of Queries (Roq)

This repository contains the code for a robust query optimization approach as presented in paper [Roq: Robust Query Optimization Based on a Risk-aware Learned Cost Model](https://arxiv.org/abs/2401.15210). This prototype is built based on `IBM Db2`. However, the concepts and ideas are directly transferable to any other relational database management system (RDBMS). This documentation explains how to use the code.

## Data Preparation
This repo assumes the dataset is already loaded in the database. Code for loading data in Db2 is available in the following repositories: 
- [IMDB](https://github.com/smakamali/setup_imdb_db2) 
- [TPCDS](https://github.com/smakamali/setup_tpcds_db2)

Steps involved in preparing data for a dataset and a workload:

1. **Installing Python dependencies:**

    Run the following command:
    ```
    pip install -r requirements.txt
    ```

2. **Collect database samples:**
    
    The first step involves collecting samples from each table of the database. These samples are used in the subsequent steps for capturing database statistics.

    Run the following command:
    ```
    python db_sampler.py
    ```

    Parameters to modify in `db_sampler.py`:

    - `schema_name` : schema name for which table samples will be collected, e.g.: `imdb`
    - `SAMPLE_SIZE` : max number of rows to be sampled from each table, e.g. `2000`


3. **Generate, encode, and label query-plan pairs:**

    Run the following command:
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

4. **Using the PyG dataset:**

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

## Running Experiments

Coming soon ...