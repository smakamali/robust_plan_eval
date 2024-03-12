import os
import numpy as np
import pandas as pd
import json
import pickle
from db_util import load_db_schema
from util import load_input_queries

def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g

def join_profile(join_list, table_datas, SAMPLE_SIZE):
    joinTypes = {}
    joinIncs = {}
    joinFactors = {}
    sampleSize = min(500,SAMPLE_SIZE) #capping the sample size at 1000, given the expensive computation for join factor and join type, inclusion factor is excluded
    print("Computing join types...")
    for idx, join in enumerate(join_list):
        print("Progress",(idx+1)/len(join_list)*100)
        # print(join)
        joinPredID = str(join[0])+'-'+str(join[1])
        joinPredIDAlt = str(join[1])+'-'+str(join[0])
        leftTab = '.'.join(join[0].split('.')[:2])
        rightTab = '.'.join(join[1].split('.')[:2])
        leftCol = join[0].split('.')[2]
        rightCol = join[1].split('.')[2]
        LDF = table_datas[leftTab].head(sampleSize).copy()
        RDF = table_datas[rightTab].head(sampleSize).copy()
        
        # compute inclusion measure
        # print("table_datas",table_datas)
        # print("leftTab",leftTab)
        # print("leftCol",leftCol)
        leftUnique = table_datas[leftTab].drop_duplicates(subset = [leftCol])
        rightUnique = table_datas[rightTab].drop_duplicates(subset = [rightCol])
        distinctMatches = pd.merge(leftUnique, rightUnique, how ='inner', 
                left_on=leftCol, right_on=rightCol).shape[0]
        minUnique = min(leftUnique.shape[0],rightUnique.shape[0])
        inclusionFactor = distinctMatches/minUnique
        joinIncs[joinPredID]= inclusionFactor
        joinIncs[joinPredIDAlt]= inclusionFactor
        
        # compute join factor
        joinFactors[joinPredID] = {}
        joinFactors[joinPredIDAlt] = {}
        LDF['key'] = 1
        RDF['key'] = 1
        crossProd = pd.merge(LDF, RDF, on ='key', suffixes = ["_l","_r"])
        cartesianProd = crossProd.shape[0]
        print(crossProd.head())
        for op in [' == ',' <= ',' >= ',' < ',' > ']:
            if op == ' == ':
                opLabel = ' = '
                joinResult = pd.merge(LDF, RDF, how ='inner', left_on=leftCol, right_on=rightCol)
            else:
                opLabel = op
                lc,rc = leftCol,rightCol
                if leftCol in RDF.columns:
                    lc = leftCol+'_l'
                if rightCol in LDF.columns:
                    rc = rightCol+'_r'
                joinResult = crossProd[(eval('crossProd.'+lc+op+'crossProd.'+rc))]
            # The following two lines need pandasql package
            # joinQ = 'SELECT * FROM LDF L INNER JOIN RDF R ON L.'+leftCol+' '+op+' R.'+rightCol+';' 
            # joinSize = sqldf(joinQ).shape[0] 
            joinSize = joinResult.shape[0]
            joinFactor = joinSize/cartesianProd
            
            joinFactors[joinPredID][opLabel] = joinFactor
            joinFactors[joinPredIDAlt][opLabel] = joinFactor

        # determine join type
        rightMatches = []
        for val in leftUnique[leftCol].values:
            rightMatches.append((table_datas[rightTab][rightCol] == val).sum())
        rightCard = max(rightMatches)
        print("rightCard",rightCard)

        leftMatches = []        
        for val in rightUnique[rightCol].values:
            leftMatches.append((table_datas[leftTab][leftCol] == val).sum())
        leftCard = max(leftMatches)
        print("leftCard",leftCard)

        if leftCard > 1 and rightCard > 1:
            joinType = 'm:n'
        elif leftCard == 1 and rightCard > 1:
            joinType = '1:n'
        elif leftCard > 1 and rightCard == 1:
            joinType = '1:n'
        elif leftCard == 1 and rightCard == 1:
            joinType = '1:1'
        elif leftCard == 0 or rightCard == 0:
            joinType = 'noMatch'
        else:
            joinType = 'other'
        
        joinTypes[joinPredID] = joinType
        joinTypes[joinPredIDAlt] = joinType

    return joinIncs, joinTypes, joinFactors

# computes the composite join factor for join with one or more join predicates
def getJoinFactor(table_datas, joinPredsDF):
    leftTab = (np.unique(joinPredsDF.TL.values))[0]
    rightTab = (np.unique(joinPredsDF.TR.values))[0]
    joinID = leftTab+'-'+rightTab
    predID = 0
    for _, row in joinPredsDF.iterrows():
        leftCol = row['CL'].split('.')[2]
        rightCol = row['CR'].split('.')[2]
        if predID == 0:
            joinResult = pd.merge(table_datas[leftTab], table_datas[rightTab], 
                how ='inner', left_on=leftCol, right_on=rightCol)
            predID+=1
        else:
            joinResult = joinResult[(joinResult[leftCol] == joinResult[rightCol])] 
            predID+=1

    joinSize = joinResult.shape[0]
    cartesianProd = table_datas[leftTab].shape[0]*table_datas[rightTab].shape[0]
    joinFactor = joinSize/cartesianProd
    return joinID, joinFactor

def combinations_of_2(l):
    for i, j in zip(*np.triu_indices(len(l), 1)):
        yield l[i], l[j]

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer

def bucketize_df(input_df, n_bins = 10, verbose = False):
    # null imputation
    print("input_df.columns",input_df.columns)
    print("input_df", input_df.shape)
    print(input_df.dtypes)
    noNullInput = input_df.copy()
    if (input_df.dtypes == 'object').any():
        onjimp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value="NaN")
        objectCols = input_df.columns[(input_df.dtypes == 'object')]
        temp1 = onjimp.fit_transform(input_df[objectCols])
        noNullInput[objectCols] = temp1
    if (input_df.dtypes != 'object').any():
        numimp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1)
        numericCols = input_df.columns[(input_df.dtypes != 'object')]
        temp2 = numimp.fit_transform(input_df[numericCols])
        noNullInput[numericCols] = temp2
    print("noNullInput",noNullInput.shape)
    # hashing
    ord_enc = OrdinalEncoder()
    hashed_data = ord_enc.fit_transform(noNullInput)
    print("hashed_data",hashed_data.shape)
    hashed_data = pd.DataFrame(hashed_data, columns = input_df.columns)
    # binning
    if verbose:
        print("binning...")
    binned_data = hashed_data.copy()
    enc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal',strategy='uniform')
    if (hashed_data.nunique() > n_bins).any():
        binned_data[hashed_data.columns[(hashed_data.nunique() > n_bins)]] = enc.fit_transform(hashed_data[hashed_data.columns[(hashed_data.nunique() > n_bins)]])

    binned_data_df = pd.DataFrame(binned_data, columns = hashed_data.columns)
    return binned_data_df

def get_chi2_matrix(input_df, verbose=False):
    df_columns = input_df.columns
    chi2_array = np.zeros((len(df_columns),len(df_columns)))
    chi2_array_inv = chi2_array.copy()
    
    for i in range(len(df_columns)):
        col1 = df_columns[i]
        for j in range(i,len(df_columns)):
            col2 = df_columns[j]
            _, p_value, _, _ = chi2_contingency(pd.crosstab(input_df[col1], input_df[col2]))
            chi2_array[i,j]=p_value
            chi2_array_inv[i,j]=1-p_value
            chi2_array[j,i]=p_value
            chi2_array_inv[j,i]=1-p_value
            if verbose:
                if p_value >= 0.05:
                    print ((col1, col2), p_value)
                else:
                    print ((col1, col2), p_value, "<==== correlated")

    chi2_matrix = pd.DataFrame(chi2_array, columns = df_columns, index= df_columns)
    chi2_matrix_inv = pd.DataFrame(chi2_array_inv, columns = df_columns, index= df_columns)
    return chi2_matrix, chi2_matrix_inv

def extractSampleInfo(schema_name, SAMPLE_SIZE, max_num_queries = 1000000, encFileID = "_id"):
    schema_name=schema_name.upper()

    with open("conn_str", "r") as conn_str_f:
        conn_str = conn_str_f.read()
    table_dict = load_db_schema(schema_name,conn_str)

    internal_dir = './internal/'
    samples_dir = './sample_data_{}_{}/'.format(schema_name.lower(),str(SAMPLE_SIZE))

    tables = list(table_dict.keys())
    print(table_dict)

    
    JoinAttractions = pd.read_csv(os.path.join(internal_dir,'JoinAttractions.csv'),header=0)

    # check if the samples dir does not exist, perform the sampling
    if not os.path.exists(samples_dir):
        from db_sampler import sampleData
        sampleData(schema_name = schema_name, # schema name
                    SAMPLE_SIZE = 2000)

    # load sample data
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

    predPerTab_df = pd.DataFrame(columns=tables)

    gini_coef_file = open(internal_dir+'gini_coef_' +str(encFileID)+'.json', 'w')
    giniCoefDict = {}

    cardToCol_file = open(internal_dir+'cardToCol_' +str(encFileID)+'.json', 'w')
    cardToColDict = {}

    correlations_file = open(internal_dir+'correlations_' +str(encFileID)+'.json', 'w')
    correlationsDict = {}

    predCountDF = pd.DataFrame(columns = ["lpCount", "ddPreds", "multiPreds", "cyclic", "selfJoin", "eqJoinPreds", "rangeJoinPreds"])
    predCountDF.index.name = "query_id"

    # get inclusion measures and join types for all joins and store as json files
    joinIncsDict, joinTypesDict, joinFatorsDict = join_profile(join_list, table_datas, SAMPLE_SIZE)

    with open(os.path.join(internal_dir,'joinIncs_{}.json'.format(str(encFileID))), 'w') as f:
        json.dump(joinIncsDict, f)
    with open(os.path.join(internal_dir,'joinTypes_{}.json'.format(str(encFileID))), 'w') as f:
        json.dump(joinTypesDict, f)
    with open(os.path.join(internal_dir,'joinFactors_{}.json'.format(str(encFileID))), 'w') as f:
        json.dump(joinFatorsDict, f)

    # get correlation matrix for all tables
    chai2matrixDict = {}
    for table in tables:
        print("computing chi2matrix for table: ",table)
        _,chai2matrix = get_chi2_matrix(bucketize_df(table_datas[table])) 
        chai2matrixDict[table] = chai2matrix
        
    # chai2matrixDict is stored as a part of db stats
    with open(os.path.join(internal_dir,'chai2matrixDict_{}.pickle'.format(str(encFileID))), 'wb') as f:
        pickle.dump(chai2matrixDict, f)
    
        
    ################# Per-query stats extraction ################

    from extract_join_attraction import get_query_join_preds
    
    # load input queries
    input_dir = './input'
    queries, query_ids = load_input_queries(input_dir)

    query_counter = 0
    for idx,sql in enumerate(queries):
        query_id = query_ids[idx].split('.')[0]
        giniCoefDict[query_id]={}
        cardToColDict[query_id]={}
        correlationsDict[query_id]={}
        multiPredj=False
        multiPred=0
        cyclic=0

        tables_dict,query_joins,local_preds,pred_cols=get_query_join_preds(schema_name,sql,verbose=True)

        # create a dataframe containing query join preds
        query_joins=np.array(query_joins)
        jpDF = pd.DataFrame(query_joins, columns = ["SL","TL","CL","SR","TR","CR","OP"])

        # capture existence of cyclic joins
        if len(jpDF.groupby(["TL","TR"])) >= len(tables_dict):
            cyclic = 1

        # capture number of self-joins
        selfJoin = jpDF[jpDF.TL == jpDF.TR].shape[0]
        
        # capture number of equality and range join preds
        eqJPred = jpDF[jpDF.OP == " = "].shape[0]
        rngJPred = jpDF[jpDF.OP != " = "].shape[0]

        # capture pairwise correlations
        predColDF=pd.DataFrame(pred_cols,columns=['tabs','cols'])
        for tab in np.unique(predColDF.tabs):
            tabCols = predColDF[predColDF.tabs == tab].cols.tolist()
            correlationsDict[query_id][tab] = {}
            if len(tabCols) == 1:
                correlationsDict[query_id][tab][tabCols[0]] = 1 
            else:
                colCombs = list(combinations_of_2(tabCols))
                for comb in colCombs:
                    correlationsDict[query_id][tab][str(comb[0])+'-'+str(comb[1])] = chai2matrixDict[tab].loc[comb[0],comb[1]]

        # capture number of predicates per table
        pred_per_tab = predColDF.groupby(by='tabs').count()
        # print("pred_per_tab",pred_per_tab)
        for table in pred_per_tab.index:
            predPerTab_df.loc[query_id,table]=pred_per_tab.loc[table,'cols']
        # print(predPerTab_df)
        # break

        jColGroups = []
        for tr in jpDF.TR.unique():
            for tl in jpDF[jpDF.TR == tr].TL.unique():
                left_col_per_join = np.unique(jpDF[(jpDF.TL == tl) & (jpDF.TR == tr)].CL.values).tolist()
                right_col_per_join = np.unique(jpDF[(jpDF.TL == tl) & (jpDF.TR == tr)].CR.values).tolist()
                jColGroups.append([left_col_per_join,right_col_per_join])
        print("jColGroups",jColGroups)
        for jcgs in jColGroups:
            multiPredj = False
            for jcg in jcgs:
                jCols = [col.split(".")[2] for col in jcg] # without schema and table
                # print(">> jCols:", jCols)
                tab = '.'.join(jcg[0].split(".")[:2])
                td = table_datas[tab]
                num_rows = td.shape[0]
                if num_rows == 0:
                    gini_coef = 0
                else:
                    # compute gini coefficient after applying local preds
                    gini_input = td[jCols].dropna()
                    gini_input['freq'] = gini_input.groupby(jCols)[jCols[0]].transform('count')
                    if gini_input.shape[0] < 1000:
                        gini_coef=gini(gini_input['freq'].values)
                    else:
                        gini_coef=gini(gini_input['freq'].sample(n=1000).values)

                # compute card to col card ratio for the join column
                if table_datas[tab].shape[0] == 0:
                    cardToCol = 0
                else:
                    cardToCol = table_datas[tab][jCols].drop_duplicates().shape[0]/table_datas[tab].shape[0]
                
                giniCoefDict[query_id][str(jcg)] = gini_coef
                cardToColDict[query_id][str(jcg)] = cardToCol

                if len(jcg) > 1:
                    multiPredj=True
            if multiPredj:
                multiPred+=1

        predCountDF.loc[query_id, "lpCount"] = len(local_preds)
        predCountDF.loc[query_id, "multiPreds"] = multiPred
        predCountDF.loc[query_id, "cyclic"] = cyclic
        predCountDF.loc[query_id, "selfJoin"] = selfJoin
        predCountDF.loc[query_id, "eqJoinPreds"] = eqJPred
        predCountDF.loc[query_id, "rangeJoinPreds"] = rngJPred

        print(" * * * * * * extractSampleInfo - Query#" + str(query_counter) +':'+ str(query_id) + " Finished! * * * * * * \n")
        query_counter += 1
        if query_counter >= max_num_queries:
            break
    json.dump(correlationsDict,correlations_file)            
    json.dump(giniCoefDict, gini_coef_file)
    json.dump(cardToColDict, cardToCol_file)
    predCountDF.to_csv(internal_dir+'pred_count_'+str(encFileID)+'.csv')
    predPerTab_df.to_csv(os.path.join(internal_dir, 'predPerTable_{}.csv'.format(encFileID)))

    correlations_file.close()
    gini_coef_file.close()
    cardToCol_file.close()
    # predPerTab_file.close()

if __name__ == '__main__':
    extractSampleInfo(schema_name = "imdb", # schema name
                        SAMPLE_SIZE = 2000, # the size of the samples to be used
                        max_num_queries = 114, # Specify the max number of queries to process
                        encFileID="job")