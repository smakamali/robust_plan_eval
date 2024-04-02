import os
import numpy as np
import pandas as pd
from util.query_parser import parse_query
from util.db_util import load_pkfk
from util.util import load_queries_from_folder

def map_op(op):
    mapping = {
        'EQ':' = ',
        'GT':' > ',
        'LT':' < ',
        'GTE':' >= ',
        'LTE':' <= ',
    }
    return mapping[op]

def complete_col(item,schema_name):
    if len(item) == 3: # if schema is specified 
        return item
    if len(item) == 2: # if schema is NOT specified
        return [schema_name]+item
    if len(item) == 1:
        raise Exception("Column qualifier must include a Table name.")

def replace_alias(col,tables_dict):
    col_copy = col.copy()
    if col_copy[1] in tables_dict:
        col_copy[1]= tables_dict[col_copy[1]]
    return col_copy

def enrich_col(col):
    return [col[0],col[1],'.'.join(col)]

def get_query_join_preds(schema_name,sql,verbose=False):

    tables_dict,join_preds,local_preds,pred_cols=parse_query(sql)
    
    schema_name=schema_name.upper()

    query_joins = []
    for join in join_preds:
        j_list = []
        for idx,item in enumerate(join):
            if idx < 2: # if item is column
                item = complete_col(item,schema_name)
                item = replace_alias(item,tables_dict)
                item = enrich_col(item)
                j_list.extend(item)
            if idx == 2: # if item is operator
                j_list.extend([map_op(item[0])])
        query_joins.append(j_list)

    l = []
    for col in pred_cols:
        col = complete_col(col,schema_name)
        col = replace_alias(col,tables_dict)
        l.append(['.'.join(col[:2]),col[2]])
    pred_cols = l

    if verbose:
        
        print("Tables {<alias>:<table_name>}:")
        print(tables_dict)

        print("Join Predicate:")
        print(join_preds)

        print("Join Predicate List:")
        print(query_joins)
        
        print("Local Predicate:")
        print(local_preds)

        print("Predicate Columns")
        print(pred_cols)

    return tables_dict, query_joins, local_preds, pred_cols

def get_all_join_attr(
        schema_name, encFileID, queries_ids,
        ):
    schema_name=schema_name.upper()
    queries, query_ids = queries_ids

    with open("conn_str", "r") as conn_str_f:
        conn_str = conn_str_f.read()

    # load referentil integrity joins
    JoinAttractions = load_pkfk(schema_name,conn_str)
    JoinAttractions.columns=["left_sch","left_tab","left_col","right_sch","right_tab","right_col"]
    JoinAttractions['op'] = ' = '

    internal_dir = './internal'

    all_joins = []
    for idx,sql in enumerate(queries):
        try:
            print("Parsing query {}".format(query_ids[idx]))
            _,joins_preds,_,_=get_query_join_preds(schema_name,sql,verbose=True)
            all_joins.extend(joins_preds)
        except:
            print("Warning: Parsing query {} failed!".format(query_ids[idx]))
            pass

    all_joins=np.array(all_joins)
    # all_joins=np.unique(all_joins,axis=0)
    # print("all_joins",all_joins)
    wl_joins_df=pd.DataFrame(all_joins, columns=JoinAttractions.columns)
    JoinAttractions = pd.concat([wl_joins_df, JoinAttractions], ignore_index=True, sort=False)
    JoinAttractions.drop_duplicates(inplace=True,keep='first')

    JoinAttractions.to_csv(os.path.join(internal_dir,'JoinAttractions_{}.csv'.format(encFileID)),index=False,header=True)
