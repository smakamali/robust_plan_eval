import os
from subprocess import Popen, PIPE
import pandas as pd
from db_util import *

def get_card_sel(conn_str):

    ibm_db_conn, ibm_db_dbi_conn = connect_to_db(conn_str, verbose=False)

    exp_sql = """
    SELECT EO.OPERATOR_ID, 
        EO.OPERATOR_TYPE,
        ESI.STREAM_COUNT AS INPUT_CARD, 
        ESO.STREAM_COUNT AS OUTPUT_CARD, 
        EO.TOTAL_COST, 
        ESI.OBJECT_NAME AS INPUT_OBJECT
    FROM SYSTOOLS.EXPLAIN_OPERATOR AS EO 
        LEFT OUTER JOIN SYSTOOLS.EXPLAIN_STREAM AS ESI 
        ON ESI.TARGET_ID = EO.OPERATOR_ID
        LEFT OUTER JOIN SYSTOOLS.EXPLAIN_STREAM AS ESO 
        ON ESO.SOURCE_ID = EO.OPERATOR_ID
    WHERE EO.OPERATOR_TYPE IN ('TBSCAN','FETCH')
        AND ESI.OBJECT_NAME <> '-';
        """

    exp_res = pd.read_sql(exp_sql,ibm_db_dbi_conn)
    exp_res["SELECTIVITY"] = exp_res.OUTPUT_CARD/exp_res.INPUT_CARD
    
    _ = close_connection_to_db(ibm_db_conn, verbose = False)

    return exp_res[["INPUT_OBJECT","INPUT_CARD","OUTPUT_CARD","SELECTIVITY"]]

def db2_explain(schema_name,sql,query_id,opt_plan_path,gen_exp_output=False, cmd_verbose=False):
    
    if not os.path.isdir(opt_plan_path):
        os.mkdir(opt_plan_path)
    file_path = os.path.join(opt_plan_path,'temp.sql')

    stmt= "explain plan for {}\n".format(sql)
    with open(file_path,'w') as f:
        f.write(stmt)
    
    cmd='db2 connect to {};\n'.format(schema_name)
    cmd+='db2 set current schema {};\n'.format(schema_name)
    cmd+='db2 delete from systools.explain_instance;\n'
    cmd+='db2 commit;\n'
    cmd+='db2 -tvf {};\n'.format(file_path)
    if gen_exp_output:
        exp_name = os.path.join(opt_plan_path,'query#{}.ex'.format(query_id))
        cmd+='db2exfmt -d {} -1 -o {};\n'.format(schema_name, exp_name)
    cmd+='connect reset;\n'
    
    process = Popen( "/bin/bash", shell=False,
                    universal_newlines=True,
                    stdin=PIPE, stdout=PIPE, stderr=PIPE)
    
    output,_ = process.communicate(cmd)
    
    if cmd_verbose:
        cmds = cmd.split('\n')
        for cmd in cmds:
            print('>> ',cmd)
    print(output)
    

def baseSelectivityGen(schema_name,max_num_queries=1000000,encFileID='_id'):
    schema_name=schema_name.upper()
    with open("conn_str", "r") as conn_str_f:
        conn_str = conn_str_f.read()
    # table_rows = load_tab_card(schema_name, conn_str)
    
    internalPath = './internal/'
    opt_plan_path = './temp/'
    input_dir = './input'
    schema_name = 'imdb'
    
    queries = []
    query_ids = []
    input_dir_enc = os.fsencode(input_dir)
    for file in os.listdir(input_dir_enc):
        filename = os.fsdecode(file)
        if filename.endswith(".sql"):
            query_ids.append(filename)
            with open(os.path.join(input_dir, filename)) as f:
                file_lines = f.readlines()
                file_content = []
                for line in file_lines:
                    if line.strip('\n').strip(' ') != '':
                        file_content.append(line)
                file_content=''.join(file_content)
                queries.extend(['SELECT '+query for query in file_content.upper().split('SELECT ')[1:]])
    
    table_dict = load_db_schema(schema_name = schema_name, conn_str = conn_str)
    tables = list(table_dict.keys())
    
    sel_df = pd.DataFrame(columns=tables) 
    card_df = pd.DataFrame(columns=tables) 

    query_counter = 0
    for idx,sql in enumerate(queries):

        query_id = query_ids[idx].split('.')[0]
        print(query_id)
        print(sql)
        
        # generate explain
        db2_explain(schema_name,sql,query_id,opt_plan_path,gen_exp_output=True,cmd_verbose=True)
        #get base table cardinality and selectivity from local predicates 
        res = get_card_sel(conn_str)
        print(res)
        
        for idx in range(len(res)):
            row = res.iloc[idx]
            sel_df.loc[query_id,'.'.join([schema_name,row.INPUT_OBJECT])] = row.SELECTIVITY
            card_df.loc[query_id,'.'.join([schema_name,row.INPUT_OBJECT])] = row.OUTPUT_CARD

        query_counter+=1
        if query_counter >= max_num_queries:
            break
    
    sel_df.to_csv(os.path.join(internalPath,'baseSel_{}.csv'.format(encFileID)))
    card_df.to_csv(os.path.join(internalPath,'baseCard_{}.csv'.format(encFileID)))
    
    # print(load_tab_card(schema_name, conn_str))

if __name__ == '__main__':
    baseSelectivityGen(schema_name = "imdb",
                        max_num_queries=500,
                        encFileID='id')