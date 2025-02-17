import os
import re
import json
import pickle
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer

models_path = os.path.join('.','lightning_models')
config_dir = os.path.join('.','config')

def load_best_model_paths(architecture_p,experiment_id,models_path=models_path):
    with open(
        os.path.join(models_path,'best_model_paths_{}_{}.pkl'.format(architecture_p,experiment_id)), 'rb') as f:
        return pickle.load(f)

def load_model_params(arch,config_dir=config_dir):
    with open(
        os.path.join(config_dir,'model_params.cfg'), 'r') as f:
        model_params = json.load(f)
    
    return model_params[arch]

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_queries_from_folder(input_dir):
    query_ids, queries = [], []
    query_ids, queries = recurse_subfolders(input_dir,query_ids,queries)

    return queries, query_ids


def recurse_subfolders(input_dir,query_ids,queries):
    
    input_dir_enc = os.fsencode(input_dir)
    
    for file in os.listdir(input_dir_enc):
        
        filename = os.fsdecode(file)
        filedir = os.path.join(input_dir,filename)
        
        if os.path.isdir(filedir):
            print("enter",filedir)
            query_ids, queries = recurse_subfolders(filedir,query_ids,queries)
        elif filename.endswith(".sql"):
            print("load from", filename)
            q_ids, qs = load_queries_from_file(input_dir,filename)
            query_ids.extend(q_ids)
            queries.extend(qs)
        else:
            pass

    return query_ids, queries


def load_queries_from_file(input_dir,filename):
    
    query_ids = [filename.split('.')[0]]
    queries = []
    
    with open(os.path.join(input_dir, filename)) as f:
        
        file_lines = f.readlines()
        file_content = []
        
        for line in file_lines:
            if line.strip('\n').strip(' ') != '':
                file_content.append(line)
        
        file_content=''.join(file_content)
        
        queries.extend(['SELECT '+query.strip(';')+';' for query in re.split('SELECT|select',file_content)[1:]])
    
    if len(query_ids) != len(queries):
        query_ids = ['q{}'.format(i) for i in range(len(queries))]
    
    return query_ids, queries

def load_input_queries(input_dir):
    queries = []
    query_ids = []
    input_dir_enc = os.fsencode(input_dir)
    for file in os.listdir(input_dir_enc):
        filename = os.fsdecode(file)
        if filename.endswith(".sql"):
            qid = filename.split('.')[0]
            query_ids.append(qid)
            with open(os.path.join(input_dir, filename)) as f:
                file_lines = f.readlines()
                file_content = []
                for line in file_lines:
                    if line.strip('\n').strip(' ') != '':
                        file_content.append(line)
                file_content=''.join(file_content)
                queries.extend(['SELECT '+query for query in re.split('SELECT |select ',file_content)[1:]])
    
    if len(query_ids) != len(queries):
        query_ids = ['q{}'.format(i) for i in range(len(queries))]

    return queries, query_ids

def find_between(s, first, last ):
    # Helper function that returns a substring between two given strings (first and last) in a larger string (s).
    try:
        start = s.index( first )
        end = s.index( last ) + len(last)
        return s[start:end]
    except ValueError:
        return ""
    
def exclusive_between(string, start, end):
    try:
        idx1 = string.index(start)
        idx2 = string.index(end)
        result = string[idx1 + len(start) : idx2]
        return result
    except ValueError:
        return ""

def join_profile(join_list, table_datas, SAMPLE_SIZE):
    joinTypes = {}
    joinIncs = {}
    joinFactors = {}
    sampleSize = min(500,SAMPLE_SIZE) #capping the sample size at 1000, given the expensive computation for join factor and join type, inclusion factor is excluded
    print("Computing join types...")
    for idx, join in enumerate(join_list):
        print("Progress",(idx+1)/len(join_list)*100)
        # print("join -------------- >",join)
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

