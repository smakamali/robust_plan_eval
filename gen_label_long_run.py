############### ReCOMPILE AND LABEL  ###############
########## TIMED OUT PLANS WITH A LARGER ###########
################ TIME OUT THRESHOLD ################

import os
import shutil
import subprocess as sp
import pickle
import math
import time
import copy
from util.db_util import connect_to_db,close_connection_to_db

hintsets=[
    [], # represents optimizer's plan using no hintsets - always keep this at the top of the list
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

def gen_label_plans(schema_name, encFileID, conn_str_path, opt_plan_path,labeled_data_dir='./labeled_data',old_timeout_thr=30, new_timeout_thr=120, dynamic_timeout=False, dynamic_timeout_factor=5):
    
    tic = time.time()

    # define input and output files
    input_path = os.path.join(labeled_data_dir,'labeled_query_plans_{}.pickle'.format(encFileID))
    output_path = os.path.join(labeled_data_dir,'labeled_query_plans_{}_pluslongrun.pickle'.format(encFileID))

    # load the connection string
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

    if not os.path.exists(labeled_data_dir):
        os.mkdir(labeled_data_dir)
        print("\n" + labeled_data_dir + " Created...\n")

    # establish connection to db
    ibm_db_conn, ibm_db_dbi_conn = connect_to_db(conn_str=conn_str)

    with open(input_path,'rb') as f:
        query_list = pickle.load(f)
    
    count_none = 0
    count_to = 0
    query_success_id = 0
    timed_out_queries = []

    new_query_list = []
    for idx,query in enumerate(query_list):
        print(query.q_id)
        query_success = False
        
        # if default plan has timed out re-execute plans with new timeout threshold
        if query.plans[0].latency > old_timeout_thr:
            count_to+=1
            
            # Collect latency for all plans
            for hintset_id, hintset in enumerate(hintsets):
                print(hintset_id)

                # compile the query using the supplied hint set
                query.compile(hintset,hintset_id,gen_exp_output=False)
                
                # execute the query with the default plan
                if hintset_id == 0:
                    errorMsg = query.execute(
                        hintset=hintset,
                        hintset_id=hintset_id, 
                        ibm_db_conn=ibm_db_conn,timeout_thr=new_timeout_thr,
                        exec_verbose = True
                        )
                opt_latency = query.plans[0].latency

                # count as successful if it runs less than the new timeout threshold
                if opt_latency < new_timeout_thr:
                    query_success = True
                    query_success_id+=1
                else:
                    break

                # update the timeout threshold if `dynamic_timeout = True`
                if dynamic_timeout:
                    timeout_thr = math.ceil(max(new_timeout_thr,opt_latency*dynamic_timeout_factor))

                # execute non-defaul plans
                if hintset_id > 1:
                    errorMsg = query.execute(
                        hintset=hintset,
                        hintset_id=hintset_id,
                        ibm_db_conn=ibm_db_conn,timeout_thr=timeout_thr,
                        exec_verbose = True
                        )
                
                # encode the generated plan
                query.plans[hintset_id].encode()
                # plan = query.plans[hintset_id]
                # plan.encode()
                # plan.plan_tree.print()
                # query.plans[hintset_id] = plan
            
            if query_success:
                new_query_list.append(query)
        else:
            new_query_list.append(query)
            

    rc = close_connection_to_db(ibm_db_conn)
    
    print("Writing to ", output_path)
    with open(output_path,'wb') as f:
        pickle.dump(new_query_list, f)

    toc = time.time()

    print("Queries with timed-out default plan: ",len(timed_out_queries))
    print("All timed-out plans", count_to)
    # print("Plans with latency == None", count_none)
    print("Time to generate and label per query: {} seconds".format((toc-tic)/count_to))


if __name__ == '__main__':
    gen_label_plans(
        schema_name = 'imdb', # schema name
        encFileID = "job_syn_all", # a unique id for the dataset
        conn_str_path = './conn_str', # path to the file containing a connection string to the database
        opt_plan_path = './job_syn_longrun_plans/', # the path used to store explain outputs and guidelines
        labeled_data_dir = './labeled_data',
        old_timeout_thr = 30, # previous timeout threshold used to avoid long running query/plans 
        new_timeout_thr = 60, # new timeout threshold
        dynamic_timeout = True, # determines whether dynamic timeout is used 
        dynamic_timeout_factor = 3 # determines the multiplier for the dynamic timeout with respect to the optimizer's plan as a baseline, used only when `dynamic_timeout = True`
    )