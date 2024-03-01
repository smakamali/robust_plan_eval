import os
import pandas as pd
import shutil
from db_util import load_db_schema,connect_to_db,close_connection_to_db

def sampleData(schema_name, SAMPLE_SIZE):
    target_path = "./sample_data_"+schema_name.lower()+"_"+str(SAMPLE_SIZE)

    with open("conn_str", "r") as conn_str_f:
        conn_str = conn_str_f.read()
        
    table_dict=load_db_schema(schema_name,conn_str)

    print("table_dict",table_dict)

    tables = list(table_dict.keys())

    if not os.path.exists(target_path):
        os.mkdir(target_path)
        print("\n" + target_path + " Created...\n")
    else:
        shutil.rmtree(target_path)
        os.mkdir(target_path)
        print("\n" + target_path + " Deleted and created again......\n")

    ibm_db_conn, ibm_db_dbi_conn = connect_to_db(conn_str, verbose=True)

    for table in tables:
        sql = "select * from "+schema_name+"."+table+" order by rand() fetch first "+str(SAMPLE_SIZE)+" rows only;"
        sample_df = pd.read_sql(sql,ibm_db_dbi_conn)
        print(sample_df.head())
        
        sample_df.to_csv(r'' + target_path + '/' + table + '_sample.csv', index = None, header=True)
    
    _ = close_connection_to_db(ibm_db_conn)

if __name__ == '__main__':
    sampleData(schema_name = "imdb", # schema name
                    SAMPLE_SIZE = 2000) # max number of rows to be sampled from each table