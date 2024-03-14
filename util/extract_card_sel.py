import os
import pandas as pd
from util.db_util import *
from util.util import load_input_queries

def baseSelectivityGen(schema_name,max_num_queries=1000000,encFileID='_id'):
    schema_name=schema_name.upper()
    with open("conn_str", "r") as conn_str_f:
        conn_str = conn_str_f.read()
    
    internalPath = './internal/'
    opt_plan_path = './temp/'
    input_dir = './input'
    
    # load input queries
    queries, query_ids = load_input_queries(input_dir)
    
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
        _,_,_=db2_explain(schema_name,sql,query_id,opt_plan_path=opt_plan_path,gen_exp_output=False,gen_guideline=False, return_cost=False,cmd_verbose=True, conn_str=conn_str)
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
    
if __name__ == '__main__':
    baseSelectivityGen(schema_name = "imdb",
                        max_num_queries=114,
                        encFileID='job')