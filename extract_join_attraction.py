import os
import numpy as np
import pandas as pd
from query_parser import parse_query
from db_util import load_pkfk

input_dir = './input'
schema_name = 'imdb'
schema_name=schema_name.upper()
with open("conn_str", "r") as conn_str_f:
    conn_str = conn_str_f.read()

JoinAttractions = load_pkfk(schema_name,conn_str)
JoinAttractions['OPERATOR'] = ' = '

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

internal_dir = './internal'
all_joins = []
for idx,sql in enumerate(queries): 
    tables_dict,join_preds,local_preds=parse_query(sql)
    
    print("Query ID: ",query_ids[idx])
    print(sql)
    
    print("Tables {<alias>:<table_name>}:")
    print(tables_dict)

    print("Join Predicate:")
    print(join_preds)

    print("Local Predicate:")
    print(local_preds)

    for join in join_preds:
        left_tab = join.split(' = ')[0].split('.')[0]
        left_col=join.split(' = ')[0].split('.')[1]
        right_tab = join.split(' = ')[1].split('.')[0]
        right_col = join.split(' = ')[1].split('.')[1]
        if left_tab in tables_dict:
            full_left = tables_dict[left_tab]
        if right_tab in tables_dict:
            full_right = tables_dict[right_tab]
        # full_join_pred = full_left+'.'+left_col+' = '+full_right+'.'+right_col
        all_joins.append([schema_name,full_left,left_col,schema_name,full_right,right_col,' = '])

all_joins=np.array(all_joins)
all_joins=np.unique(all_joins,axis=0)
wl_joins_df=pd.DataFrame(all_joins, columns=JoinAttractions.columns)
JoinAttractions = pd.concat([wl_joins_df, JoinAttractions], ignore_index=True, sort=False)
JoinAttractions.drop_duplicates(inplace=True)

JoinAttractions.to_csv(os.path.join(internal_dir,'JoinAttractions.csv'),index=False,header=False)