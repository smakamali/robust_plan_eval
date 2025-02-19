import os
import json
import pickle
import shutil
import pandas as pd
from util.db_util import *
from util.util import *
from util.extract_join_attraction import get_all_join_attr

def sampleData(schema_name, SAMPLE_SIZE, materialize=False):
    samples_dir = './sample_data_{}_{}/'.format(schema_name.lower(),str(SAMPLE_SIZE))

    if materialize:
            
        if not os.path.exists(samples_dir):
            os.mkdir(samples_dir)
            print("\n" + samples_dir + " Created...\n")
        else:
            shutil.rmtree(samples_dir)
            os.mkdir(samples_dir)
            print("\n" + samples_dir + " Deleted and created again......\n")

    with open("conn_str", "r") as conn_str_f:
        conn_str = conn_str_f.read()

    ibm_db_conn, ibm_db_dbi_conn = connect_to_db(conn_str, verbose=True)

    table_dict=load_db_schema(schema_name,conn_str)

    print("table_dict",table_dict)

    tables = list(table_dict.keys())

    table_datas={}
    for table in tables:
        sql = "select * from "+schema_name+"."+table+" order by rand() fetch first "+str(SAMPLE_SIZE)+" rows only;"
        sample_df = pd.read_sql(sql,ibm_db_dbi_conn)
        print(sample_df.head())
        table_datas[table] = sample_df

        if materialize:
            
            tab_path = os.path.join(samples_dir,'{}_sample.csv'.format(table))
            sample_df.to_csv(tab_path, index = None, header=True)
    
    _ = close_connection_to_db(ibm_db_conn)

    return table_datas


def get_db_stats(
        schema_name, 
        queries_ids,
        internal_dir='./internal', SAMPLE_SIZE=2000, encFileID='id'
        ):

    with open("conn_str", "r") as conn_str_f:
        conn_str = conn_str_f.read()
    table_dict = load_db_schema(schema_name,conn_str)
    tables = list(table_dict.keys())

    # generate and extract join attractions
    get_all_join_attr(
        schema_name, encFileID ,
        queries_ids,
        )
    JoinAttractions = pd.read_csv(os.path.join(internal_dir,'JoinAttractions_{}.csv'.format(encFileID)),header=0)

    # samples_dir = './sample_data_{}_{}/'.format(schema_name.lower(),str(SAMPLE_SIZE))

    # table_datas = {}
    # for table in tables:
    #     print(table)
    #     # Read csv files with "\" as escapechar and """ as quotechar. 
    #     tab_path = os.path.join(samples_dir,'{}_sample.csv'.format(table))
    #     table_datas[table] = pd.read_csv(tab_path, escapechar="\\", index_col=False, header=0)

    table_datas = sampleData(schema_name = schema_name, SAMPLE_SIZE = SAMPLE_SIZE, materialize=True)

    join_list = []
    for idx in range(len(JoinAttractions)):
        joinAttr = JoinAttractions.loc[idx]
        left_col = joinAttr.left_col
        right_col = joinAttr.right_col
        join_list.append([left_col, right_col])

    # get inclusion measures and join types for all joins and store as json files
    joinIncsDict, joinTypesDict, joinFatorsDict = join_profile(join_list, table_datas, SAMPLE_SIZE)

    # get correlation matrix for all tables
    chai2matrixDict = {}
    for table in tables:
        print("computing chi2matrix for table: ",table)
        if table_datas[table].shape[0] > 0:
            _,chai2matrix = get_chi2_matrix(bucketize_df(table_datas[table]))
        else:
            _,chai2matrix = None, None
        chai2matrixDict[table] = chai2matrix

        
    # the db stats are stored to disk
    with open(os.path.join(internal_dir,'joinIncs_{}.json'.format(str(encFileID))), 'w') as f:
        json.dump(joinIncsDict, f)
    with open(os.path.join(internal_dir,'joinTypes_{}.json'.format(str(encFileID))), 'w') as f:
        json.dump(joinTypesDict, f)
    with open(os.path.join(internal_dir,'joinFactors_{}.json'.format(str(encFileID))), 'w') as f:
        json.dump(joinFatorsDict, f)
    with open(os.path.join(internal_dir,'chai2matrixDict_{}.pickle'.format(str(encFileID))), 'wb') as f:
        pickle.dump(chai2matrixDict, f)
