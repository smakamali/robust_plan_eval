import os
import numpy as np
import pandas as pd
from query_parser import parse_query
from db_util import load_pkfk
from util import load_input_queries



def get_query_join_preds(schema_name,sql,verbose=False):
    tables_dict,join_preds,local_preds,pred_cols=parse_query(sql)
    
    schema_name=schema_name.upper()

    # TODO: The following split should be done in parse_query
    query_joins = []
    for join in join_preds:
        left_tab = join.split(' = ')[0].split('.')[0]
        right_tab = join.split(' = ')[1].split('.')[0]
        full_left = left_tab
        if left_tab in tables_dict:
            full_left = tables_dict[left_tab]
        full_right = right_tab
        if right_tab in tables_dict:
            full_right = tables_dict[right_tab]
        
        left_col=schema_name+'.'+full_left+'.'+join.split(' = ')[0].split('.')[1]
        right_col = schema_name+'.'+full_right+'.'+join.split(' = ')[1].split('.')[1]
        
        query_joins.append([schema_name,full_left,left_col,schema_name,full_right,right_col,' = '])

    pred_cols_full = []
    for col in pred_cols:
        tab = col.split('.')[0]
        col = col.split('.')[1]
        if tab in tables_dict:
            full_tab = tables_dict[tab]
        else:
            full_tab = tab
        
        full_column=[schema_name+'.'+full_tab,col]
        
        pred_cols_full.append(full_column)
    pred_cols = pred_cols_full

    if verbose:
        # print("Query ID: ",query_ids[idx])
        # print(sql)
        
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

def get_all_join_attr(schema_name):

    schema_name=schema_name.upper()

    with open("conn_str", "r") as conn_str_f:
        conn_str = conn_str_f.read()

    # load referentil integrity joins
    JoinAttractions = load_pkfk(schema_name,conn_str)
    JoinAttractions.columns=["left_sch","left_tab","left_col","right_sch","right_tab","right_col"]
    JoinAttractions['op'] = ' = '

    # load input queries
    input_dir = './input'
    queries, query_ids = load_input_queries(input_dir)

    internal_dir = './internal'

    all_joins = []
    for idx,sql in enumerate(queries):
        print(query_ids[idx])
        # print(sql)
        tables_dict,joins_preds,local_preds,pred_cols=get_query_join_preds(schema_name,sql,verbose=False)
        all_joins.extend(joins_preds)

    all_joins=np.array(all_joins)
    # all_joins=np.unique(all_joins,axis=0)
    wl_joins_df=pd.DataFrame(all_joins, columns=JoinAttractions.columns)
    JoinAttractions = pd.concat([wl_joins_df, JoinAttractions], ignore_index=True, sort=False)
    JoinAttractions.drop_duplicates(inplace=True,keep='first')

    JoinAttractions.to_csv(os.path.join(internal_dir,'JoinAttractions.csv'),index=False,header=True)

if __name__ == '__main__':
    get_all_join_attr(schema_name='imdb')

# # THE FOLLOWING LINES FOR TESTING
# sql = """SELECT MIN(cn.name) AS movie_company,
#        MIN(mi_idx.info) AS rating,
#        MIN(t.title) AS western_violent_movie
# FROM company_name AS cn,
#      company_type AS ct,
#      info_type AS it1,
#      info_type AS it2,
#      keyword AS k,
#      kind_type AS kt,
#      movie_companies AS mc,
#      movie_info AS mi,
#      movie_info_idx AS mi_idx,
#      movie_keyword AS mk,
#      title AS t
# WHERE cn.country_code != '[us]'
#   AND it1.info = 'countries'
#   AND it2.info = 'rating'
#   AND k.keyword IN ('murder',
#                     'murder-in-title',
#                     'blood',
#                     'violence')
#   AND kt.kind IN ('movie',
#                   'episode')
#   AND mc.note NOT LIKE '%(USA)%'
#   AND mc.note LIKE '%(200%)%'
#   AND mi.info IN ('Sweden',
#                   'Norway',
#                   'Germany',
#                   'Denmark',
#                   'Swedish',
#                   'Danish',
#                   'Norwegian',
#                   'German',
#                   'USA',
#                   'American')
#   AND mi_idx.info < '8.5'
#   AND t.production_year > 2005
#   AND kt.id = t.kind_id
#   AND t.id = mi.movie_id
#   AND t.id = mk.movie_id
#   AND t.id = mi_idx.movie_id
#   AND t.id = mc.movie_id
#   AND mk.movie_id = mi.movie_id
#   AND mk.movie_id = mi_idx.movie_id
#   AND mk.movie_id = mc.movie_id
#   AND mi.movie_id = mi_idx.movie_id
#   AND mi.movie_id = mc.movie_id
#   AND mc.movie_id = mi_idx.movie_id
#   AND k.id = mk.keyword_id
#   AND it1.id = mi.info_type_id
#   AND it2.id = mi_idx.info_type_id
#   AND ct.id = mc.company_type_id
#   AND cn.id = mc.company_id;
# """
# tables_dict,all_joins,local_preds,pred_cols=get_query_join_preds('imdb',sql,verbose=True)
# print(all_joins)
