############ LOAD QUERIES, COMPILE AND LABEL PLANS #############

#  TODO: automate sample collection 
#  TODO: specify a separate directory for output
# --> TODO: make extraction of join attractions from the input optional and limited to the max_num_queries -> add a db stats extraction in this script before loading queries.

import os
import shutil
import subprocess as sp
import pickle
import math
import time
from subprocess import Popen, PIPE
from util.util import load_input_queries, load_queries_from_folder
from util.base_classes import Query
from util.db_util import connect_to_db,close_connection_to_db


hintsets=[
    [], # represents optimizer's plan using no hintsets - always keep this at top of the list
    ['db2 .opt set disable nljn;\n'],
    ['db2 .opt set disable nljn;\n','db2 .opt set disable iscan;\n'],
    ['db2 .opt set disable hsjn;\n'],
    ['db2 .opt set disable hsjn;\n','db2 .opt set disable iscan;\n'],
    ['db2 .opt set disable mgjn;\n'],
    ['db2 .opt set disable mgjn;\n','db2 .opt set disable iscan;\n'],
    ['db2 .opt set disable nljn;\n', 'db2 .opt set disable mgjn;\n'],
    ['db2 .opt set disable nljn;\n', 'db2 .opt set disable mgjn;\n','db2 .opt set disable iscan;\n'],
    ['db2 .opt set disable nljn;\n', 'db2 .opt set disable hsjn;\n'],
    ['db2 .opt set disable nljn;\n', 'db2 .opt set disable hsjn;\n','db2 .opt set disable iscan;\n'],
    ['db2 .opt set disable mgjn;\n', 'db2 .opt set disable hsjn;\n'],
    ['db2 .opt set disable mgjn;\n', 'db2 .opt set disable hsjn;\n','db2 .opt set disable iscan;\n'],
    ]

def gen_label_plans(start, num_samples, data_slice, schema_name, encFileID, conn_str_path, input_dir='./input/', opt_plan_path='./optimizer_plans/', internal_dir='./internal/',labeled_data_dir='./labeled_data', sample_size=2000,timeout_thr=60, dynamic_timeout=False, dynamic_timeout_factor=5):

    tic = time.time()

    with open(conn_str_path, "r") as conn_str_f:
        conn_str = conn_str_f.read()
    conn_str = conn_str.strip('\n')+"CURRENTSCHEMA={};\n".format(schema_name)

    rc = sp.run('. ~/sqllib/db2profile', shell = True)
    rc = sp.run('~/db2cserv -d on', shell = True)
    rc = sp.run('db2stop force', shell = True)
    rc = sp.run('db2start', shell = True)

    if not os.path.exists(opt_plan_path):
        os.mkdir(opt_plan_path)
        print("\n" + opt_plan_path + " Created...\n")
    else:
        shutil.rmtree(opt_plan_path)
        os.mkdir(opt_plan_path)
        print("\n" + opt_plan_path + " Deleted and created again......\n")

    if not os.path.exists(internal_dir):
        os.mkdir(internal_dir)
        print("\n" + internal_dir + " Created...\n")

    if not os.path.exists(labeled_data_dir):
        os.mkdir(labeled_data_dir)
        print("\n" + labeled_data_dir + " Created...\n")

    output_path = os.path.join(labeled_data_dir,'labeled_query_plans_{}.pickle'.format(encFileID))

    # err_files_path = os.path.join(internal_dir,'error_files_{}'.format(encFileID))
    # with open(err_files_path, 'w') as f:
    #     f.write("Explain errors for "+ encFileID+"\n")
    # print("\n" + err_files_path + " Created...\n")

    # establish connection to db
    ibm_db_conn, ibm_db_dbi_conn = connect_to_db(conn_str=conn_str)

    # load input queries
    queries, query_ids = load_queries_from_folder(input_dir)

    # get a slice of the input queries if `start` and `num_samples` are given or if `data_slice` is given
    if data_slice is None:
        if start is not None and num_samples is not None:
            max_samples = len(queries)
            step = int(max_samples/num_samples)
            data_slice = slice(start,max_samples,step)
        else:
            data_slice = slice(0,len(queries))

    queries = queries[data_slice]
    query_ids = query_ids[data_slice]

    # collect db stats before proceeding to processing queries
    join_inc_path = os.path.join(internal_dir,'joinIncs_{}.json'.format(str(encFileID)))
    join_type_path = os.path.join(internal_dir,'joinTypes_{}.json'.format(str(encFileID)))
    join_factor_path = os.path.join(internal_dir,'joinFactors_{}.json'.format(str(encFileID)))
    chai2_matrix_path = os.path.join(internal_dir,'chai2matrixDict_{}.pickle'.format(str(encFileID)))

    with open(conn_str_path, "r") as conn_str_f:
        conn_str = conn_str_f.read()

    # db stats are only collected if they do not exist 
    if not (os.path.isfile(join_inc_path) and os.path.isfile(join_type_path) and os.path.isfile(join_factor_path) and os.path.isfile(chai2_matrix_path)):

        from util.get_db_stats import get_db_stats
        get_db_stats(
            schema_name=schema_name, 
            queries_ids=(queries, query_ids),
            # input_dir=input_dir, 
            internal_dir=internal_dir, SAMPLE_SIZE=sample_size, encFileID=encFileID
            )

    query_success_id=0
    guide_success_id=0
    query_list=[]
    for idx,sql in enumerate(queries):
        one_success = False
        try:
            q_id = query_ids[idx]
            
            # initialize the query object
            query = Query(schema_name,sql,q_id,encFileID=encFileID, conn_str_path=conn_str_path,input_dir=input_dir, opt_plan_path=opt_plan_path,internal_dir=internal_dir,sample_size=sample_size)
            
            # encode the query
            query.encode()
            
            for hintset_id, histset in enumerate(hintsets):
                
                print("compile query {} with hintset {}".format(str(idx),str(hintset_id)))
                
                try:
                    # compile the query using the supplied hint set
                    query.compile(histset,hintset_id,gen_exp_output=False)
                    
                    query.plans[hintset_id].encode()
                    
                    # execute the default plan
                    if hintset_id == 0:
                        _ = query.execute(
                            hintset=histset,
                            hintset_id=hintset_id, 
                            ibm_db_conn=ibm_db_conn,timeout_thr=timeout_thr,
                            exec_verbose = True
                            )
                    
                    # if the defaul plan times out, skip the whole query
                    opt_latency = query.plans[0].latency
                    if opt_latency > timeout_thr:
                        print("Default plan timed out. Skipping query {}...".format(q_id))
                        break
                    
                    # update the timeout threshold if `dynamic_timeout == True`
                    if dynamic_timeout:
                        new_timeout_thr = math.ceil(min(timeout_thr,opt_latency*dynamic_timeout_factor))
                    else:
                        new_timeout_thr = timeout_thr
                    
                    # execute non-defaul plans
                    if hintset_id > 0:
                        errorMsg = query.execute(hintset=histset,
                            hintset_id=hintset_id, 
                            ibm_db_conn=ibm_db_conn,timeout_thr=new_timeout_thr,
                            exec_verbose = True
                            )
                    
                    guide_success_id+=1
                    one_success = True
                
                except:
                    print("Execution failed for query {}, plan {}.".format(q_id,hintset_id))
                    pass

            if one_success:
                query_list.append(query)
                query_success_id+=1
            
        except:
            print("Encoding failed for query {}.".format(q_id))
            pass
        
        # checkpoint - write to disk every 100 query
        if query_success_id%5:
            with open(output_path, 'wb') as f:
                pickle.dump(query_list, f)

        # Unsuccessful compiles leads to creating Db2 dump file that eats up space. This code block periodically clears the dump files.
        if query_success_id%5:
            cmd = 'rm -r -f ~/sqllib/db2dump/DIAG0000/FODC_AppErr_*'
            process = Popen( "/bin/bash", shell=False, universal_newlines=True,
                        stdin=PIPE, stdout=PIPE, stderr=PIPE)
            output, _ = process.communicate(cmd)

    rc = close_connection_to_db(ibm_db_conn)


    # final write to disk
    with open(output_path,'wb') as f:
        pickle.dump(query_list, f)

    toc = time.time()

    print("********  Processing all queries is done!  ********")
    print(query_success_id, "queries were processed.")
    print(guide_success_id, "explains and guidelines were created.")
    print("Time to generate and label per query: {} seconds".format((toc-tic)/query_success_id))
    print()

if __name__ == '__main__':

    gen_label_plans(
        data_slice= None, # Specify the max number of queries to process
        start = 1, # alternative to data_slice, gives the starting index, must be provided together with `num_samples`
        num_samples = 500, # alternative to data_slice, gives the number of samples, must be provided together with `start`
        schema_name = 'imdb', # schema name
        encFileID = "ceb-13k-500-1", # a unique id for the dataset
        conn_str_path = './conn_str', # path to the file containing a connection string to the database
        input_dir = "./input/ceb-imdb-13k/", # the directory that contains query.sql file(s)
        opt_plan_path = './job_ceb-13k-1_plans/', # the path used to store explain outputs and guidelines
        internal_dir = './internal/', # the path to store intermediary files
        labeled_data_dir = './labeled_data/',
        sample_size = 2000, # number of samples used per table
        timeout_thr = 60, # timeout threshold to avoid long running query/plans 
        dynamic_timeout = True, # determines whether dynamic timeout is used 
        dynamic_timeout_factor = 10 # determines the multiplier for the dynamic timeout with respect to the optimizer's plan as a baseline, used only when `dynamic_timeout = True`
        )

    