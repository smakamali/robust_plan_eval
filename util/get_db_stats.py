import json
import pickle
from util.db_util import *
from util.util import *
from util.extract_join_attraction import get_all_join_attr

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

    samples_dir = './sample_data_{}_{}/'.format(schema_name.lower(),str(SAMPLE_SIZE))

    table_datas = {}
    for table in tables:
        print(table)
        # Read csv files with "\" as escapechar and """ as quotechar. 
        tab_path = os.path.join(samples_dir,'{}_sample.csv'.format(table))
        table_datas[table] = pd.read_csv(tab_path, escapechar="\\", index_col=False, header=0)

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
        if table_datas[table].shape[0] > 1:
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
