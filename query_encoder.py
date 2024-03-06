# what is needed? 
#   - the adjacency matrix
#   - join type
#   - join operator
#   - column cardinalities -> colcard / card
#   - table cardinalities -> colcard / card
#   - column skewness
#   - join cardinality from sample -> join selectivity
#   - distinct values from each side of a join
#   - number of matches between the two sides -> inclusion factor
#   - local predicate selectivites (from either the sample or the optimizer)
#   - pairwise correlation between predicate columns
import os
import json
import numpy as np
import pandas as pd
from extract_join_attraction import get_query_join_preds
from db_util import load_tab_card

def encodeOps (ops):
    encOps = []
    for op in [" = ", " <= ", " >= ", " < ", " > "]:
        if op in ops:
            encOps.extend([1])
        else:
            encOps.extend([0])
    return np.array(encOps)

def encodeTypes (joinTypes):
    encTypes = []
    for jt in ["m:n", "1:n", "1:1", "noMatch"]:
        if jt in joinTypes:
            encTypes.extend([1])
        else:
            encTypes.extend([0])
    return np.array(encTypes)

def getJoinProfile(joinPredsDF, joinIncsDict, joinTypesDict, joinFactorsDict):
    joinFactors = []
    joinIncFactors = []
    joinTypes = []
    for _, row in joinPredsDF.iterrows():
        leftCol = row['CL']
        rightCol = row['CR']
        joinPredID = leftCol+'-'+rightCol
        joinIncFactors.append(joinIncsDict[joinPredID])
        joinTypes.append(joinTypesDict[joinPredID])
        joinFactors.append(joinFactorsDict[joinPredID][row['OP']])
    return joinIncFactors, joinTypes, joinFactors

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def generate_join_graph(schema_name,encFileID,max_num_queries):
    
    schema_name=schema_name.upper()
    input_dir = './input/'
    internal_dir = './internal'
    enc_path = './encodings/'

    with open("conn_str", "r") as conn_str_f:
        conn_str = conn_str_f.read()

    if not os.path.exists(enc_path):
        os.mkdir(enc_path)
    torch_enc = open(os.path.join(enc_path,'query_graph_torch_enc_{}.json'.format(encFileID)), 'w')

    # join_attractions = pd.read_csv(os.path.join(internal_dir,'JoinAttractions.csv'),header=0)
    # print(join_attractions)

    table_card_dict = load_tab_card(schema_name,conn_str)
    print(table_card_dict)
    table_df = pd.DataFrame.from_dict(table_card_dict,orient='index')
    table_df.reset_index(inplace=True)
    table_df.sort_values(by=0,ascending=False,inplace=True)
    table_df.reset_index(inplace=True,drop=True)
    table_df.reset_index(inplace=True)
    table_df.columns=['tab_idx','table','card']

    print(table_df)

    id_tab={}
    for idx in range(len(table_df)):
        id_tab[table_df.iloc[idx].table] = table_df.iloc[idx].tab_idx

    print(id_tab)

    max_tables = len(id_tab)

    base_sels = pd.read_csv(os.path.join(internal_dir,'baseSel_{}.csv'.format(str(encFileID))), header=0,index_col=0)

    base_cards = pd.read_csv(os.path.join(internal_dir,'baseCard_{}.csv'.format(str(encFileID))), header=0,index_col=0)

    predPerTab = pd.read_csv(os.path.join(internal_dir,'predPerTable_{}.csv'.format(str(encFileID))), header=0,index_col=0)

    with open(os.path.join(internal_dir,'gini_coef_{}.json'.format(str(encFileID))),'r') as f:
        giniCoefDict = json.load(f)

    with open(os.path.join(internal_dir,'cardToCol_{}.json'.format(str(encFileID))),'r') as f:
        cardToColDict = json.load(f)

    with open(os.path.join(internal_dir,'correlations_{}.json'.format(str(encFileID))),'r') as f:
        correlationsDict = json.load(f)

    with open(os.path.join(internal_dir,'joinIncs_{}.json'.format(str(encFileID))), 'r') as f:
        joinIncsDict = json.load(f)

    with open(os.path.join(internal_dir,'joinTypes_{}.json'.format(str(encFileID))), 'r') as f:
        joinTypesDict = json.load(f)

    with open(os.path.join(internal_dir,'joinFactors_{}.json'.format(str(encFileID))), 'r') as f:
        joinFactorsDict = json.load(f)


    AM = np.zeros((max_tables,max_tables,5)) # initialize Adjacency Matrix
    CR = np.zeros((max_tables,max_tables,1)) # initialize Card Ratio Matrix
    CC = np.zeros((max_tables,max_tables,2)) # initialize Card to Col Card Matrix
    IF = np.zeros((max_tables,max_tables,3)) # initialize Inclusion Factor Matrix
    JT = np.zeros((max_tables,max_tables,4)) # initialize Join Type Matrix
    JF = np.zeros((max_tables,max_tables,3)) # initialize Join Factor Matrix
    SK = np.zeros((max_tables,max_tables,2)) # initialize Skewness Factor
    NC = np.zeros((max_tables,6)) # initialize Node Coordinates Matrix

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
    
    torchEncoding={}
    query_counter = 0
    for idx,sql in enumerate(queries):
        query_id = query_ids[idx].split('.')[0]
        
        print(sql)

        AM[AM!=0]=0 
        CR[CR!=0]=0
        CC[CC!=0]=0
        IF[IF!=0]=0
        JT[JT!=0]=0
        JF[JF!=0]=0
        SK[SK!=0]=0
        NC[NC!=0]=0
        torchEncoding[query_id]={}

        tables_dict,query_joins,local_preds,pred_cols=get_query_join_preds(schema_name,sql,verbose=False)

        # get a list of tables referenced in the query
        table_list = ['.'.join([schema_name,str(tables_dict[key])]) for key in tables_dict]

        jc_list = np.array(query_joins)[:,[2,5,6]]
        print(jc_list)
    
        jpList = []
        for pred in jc_list:
            left_col, right_col, op = pred[0], pred[1], pred[2]
            left_tab = '.'.join(pred[0].split('.')[0:2])
            right_tab = '.'.join(pred[1].split('.')[0:2])
            jpList.append([left_tab,left_col,right_tab,right_col,op])
        jpArr = np.array(jpList)
        jpDF = pd.DataFrame(jpArr, columns = ["TL","CL","TR","CR","OP"])

        print(jpDF)

        jColGroups = {}
        for tr in jpDF.TR.unique():
            for tl in jpDF[jpDF.TR == tr].TL.unique():
                jColGroups[tl+'-'+tr] = [
                    np.unique(jpDF[(jpDF.TL == tl) & (jpDF.TR == tr)].CL.values).tolist(),
                    np.unique(jpDF[(jpDF.TL == tl) & (jpDF.TR == tr)].CR.values).tolist(),
                    jpDF[(jpDF.TL == tl) & (jpDF.TR == tr)].OP.values.tolist()
                    ]
        print(">> jColGroups: ",jColGroups)

        for pred in jColGroups.keys():
            left_tab, right_tab = pred.split('-')
            left_col, right_col, predOp = jColGroups[pred][0], jColGroups[pred][1], jColGroups[pred][2]
            # Populate Adjacency Matrix
            AM[id_tab[left_tab],id_tab[right_tab]] = encodeOps(predOp)
            AM[id_tab[right_tab],id_tab[left_tab]] = encodeOps(predOp)

            # Populate Card Ratio Matrix
            card_ratio = max([base_cards.at[query_id, left_tab], base_cards.at[query_id, right_tab]])/(min([base_cards.at[query_id, left_tab], base_cards.at[query_id, right_tab]])+1)
            # card_ratio = base_cards.at[query_id, left_tab.split(".")[1]] / (base_cards.at[query_id, right_tab.split(".")[1]]+1)
            CR[id_tab[left_tab],id_tab[right_tab]] = card_ratio
            # CR[id_tab[right_tab],id_tab[left_tab]] = card_ratio
            CR[id_tab[right_tab],id_tab[left_tab]] = card_ratio

            # Populate card to col card ratio
            CC[id_tab[left_tab],id_tab[right_tab],0] = cardToColDict[str(query_id)][str(left_col)]
            CC[id_tab[left_tab],id_tab[right_tab],1] = cardToColDict[str(query_id)][str(right_col)]
            CC[id_tab[right_tab],id_tab[left_tab],0] = cardToColDict[str(query_id)][str(left_col)]
            CC[id_tab[right_tab],id_tab[left_tab],1] = cardToColDict[str(query_id)][str(right_col)]

            # Get Join Profile Info
            joinPredsDF = jpDF[(jpDF.TL == left_tab) & (jpDF.TR == right_tab)]
            joinIncFactors, joinTypes, joinFactors = getJoinProfile(joinPredsDF, joinIncsDict, joinTypesDict, joinFactorsDict)

            # populate join inclusion factor
            IF[id_tab[left_tab],id_tab[right_tab],0] = min(joinIncFactors)
            IF[id_tab[left_tab],id_tab[right_tab],1] = sum(joinIncFactors)/len(joinIncFactors)
            IF[id_tab[left_tab],id_tab[right_tab],2] = max(joinIncFactors)
            IF[id_tab[right_tab],id_tab[left_tab],0] = min(joinIncFactors)
            IF[id_tab[right_tab],id_tab[left_tab],1] = sum(joinIncFactors)/len(joinIncFactors)
            IF[id_tab[right_tab],id_tab[left_tab],2] = max(joinIncFactors)

            # populate join types
            encodedTypes = encodeTypes(joinTypes)
            JT[id_tab[left_tab],id_tab[right_tab]] = encodedTypes
            JT[id_tab[right_tab],id_tab[left_tab]] = encodedTypes

            # populate join factor
            JF[id_tab[left_tab],id_tab[right_tab],0] = min(joinFactors)
            JF[id_tab[left_tab],id_tab[right_tab],1] = sum(joinFactors)/len(joinFactors)
            JF[id_tab[left_tab],id_tab[right_tab],2] = max(joinFactors)
            JF[id_tab[right_tab],id_tab[left_tab],0] = min(joinFactors)
            JF[id_tab[right_tab],id_tab[left_tab],1] = sum(joinFactors)/len(joinFactors)
            JF[id_tab[right_tab],id_tab[left_tab],2] = max(joinFactors)

            SK[id_tab[left_tab],id_tab[right_tab],0] = giniCoefDict[str(query_id)][str(left_col)]
            SK[id_tab[left_tab],id_tab[right_tab],1] = giniCoefDict[str(query_id)][str(right_col)]
            SK[id_tab[right_tab],id_tab[left_tab],0] = giniCoefDict[str(query_id)][str(left_col)]
            SK[id_tab[right_tab],id_tab[left_tab],1] = giniCoefDict[str(query_id)][str(right_col)]

        # Populate Node Coordinates Matrix
        for schema_tab in table_list:
            # tab = schema_tab.split('.')[1]
            NC[id_tab[schema_tab],0] = base_cards.at[query_id,schema_tab]
            NC[id_tab[schema_tab],1] = base_sels.at[query_id,schema_tab]
            NC[id_tab[schema_tab],2] = predPerTab.at[query_id,schema_tab]
            # encode stats on pairwise correlations between predicate columns per table
            colCorDict = correlationsDict[str(query_id)][schema_tab]
            corArr = np.array([colCorDict[comb] for comb in colCorDict.keys()])
            NC[id_tab[schema_tab],3] = corArr.min()
            NC[id_tab[schema_tab],4] = corArr.mean()
            NC[id_tab[schema_tab],5] = corArr.max()

        # organize the encoding in a format compatible with pytorch geometric
        torchEncoding[query_id]={}
        edgeFeaturesMatrix = np.concatenate((AM,CR,CC,IF,JT,JF,SK), axis=2)
        nodeFeatures = NC.copy()
        print("edgeFeaturesMatrix.shape",edgeFeaturesMatrix.shape)
        print("nodeFeatures.shape",nodeFeatures.shape)
        
        # Construct edge indices
        AdjMat = np.max(AM, axis=2)
        edgeIndc=[]
        for i in range(AdjMat.shape[0]):
            for j in range(AdjMat.shape[1]):
                if AdjMat[i,j]>0:
                    edgeIndc.append([i,j])
                # else:
                #     if edge_zero_padding:
                #         edgeIndc.append([0,0]) # padding the edgeIndc with zeros for dummy edges
                #     else:
                #         pass
                    
        edgeIndc = np.array(edgeIndc)
        print("edgeIndc",edgeIndc.shape)
        
        # Construct edge features
        edgeFeat=[]
        for i in range(edgeIndc.shape[0]):
            if edgeIndc[i].sum() != 0:
                edgeFeat.append(edgeFeaturesMatrix[edgeIndc[i][0],edgeIndc[i][1],:])
            # else:
            #     if edge_zero_padding:
            #         edgeFeat.append([0]*edgeFeaturesMatrix.shape[2]) # padding the edgeFeat with zeros for dummy edges
            #     else:
            #         pass
        edgeFeat = np.array(edgeFeat)
        print("edgeFeat",edgeFeat.shape)

        edgeIndc = np.transpose(edgeIndc)
        
        # Populate the encodings and labels in the results dictionary
        torchEncoding[query_id]['nodeFeatures'] = (nodeFeatures)
        torchEncoding[query_id]['edgeIndc'] = (edgeIndc)
        torchEncoding[query_id]['edgeFeat'] = (edgeFeat)
        # for label in ['card','cartProd','OptEst','OptEstInd']:
        #     torchEncoding[query_id][label] = (labelsDF.at[query_id,label])
        print(" * * * * * * Query # {} encoded! * * * * * * \n".format(query_id))

        query_counter+=1
        if query_counter >= max_num_queries:
            break
    json.dump(torchEncoding, torch_enc, cls=NpEncoder)
    torch_enc.close()
        
        
if __name__ == '__main__':
    generate_join_graph(schema_name='imdb',
                        encFileID='id',
                        max_num_queries=200)
