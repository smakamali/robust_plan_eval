# robust_plan_eval

## Data Preparation
Steps involved in preparing data for a workload:

1. **Collect database samples:**

    *a. Modify parameters in `db_sampler.py`*
    - `schema_name` : schema name for which table samples will be collected
    - `SAMPLE_SIZE` : max number of rows to be sampled from each table

    *b. Run `python db_sampler.py`*

2. **Generate, encode, and label samples:**

    *a. Modify parameters set in `gen_label_plans.py`.*

    -  `max_num_queries` : Specify the max number of queries to explain
    - `schema_name` : schema name
    - `encFileID` : a unique id for the dataset
    - `conn_str_path` : path to the file containing a connection string to the database
    - `input_dir` : the directory that contains query.sql file(s)
    - `opt_plan_path` : the path used to store explain outputs and guidelines
    - `internal_dir` : the path to store intermediary files
    - `sample_size` : number of samples used per table, set the same as the value used in `db_sampler.py`
    - `timeout_thr` : timeout threshold to avoid long running query/plans
    - `dynamic_timeout` : determines whether dynamic timeout is used
    - `dynamic_timeout_factor` : determines the multiplier for the dynamic timeout with respect to the optimizer's plan as a baseline, used only when `dynamic_timeout = True`

    *b. Run `python gen_label_plans.py`*
