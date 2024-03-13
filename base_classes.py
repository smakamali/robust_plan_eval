# TODO: if possible, switch to SQLAlchemy for using pandas to fetch results from Db2
# TODO: what if explain fails?

import os
import numpy as np
import pandas as pd
import json
import pickle
from extract_join_attraction import get_query_join_preds
from db_util import db2_explain, db2_execute, get_card_sel, load_db_schema
from extract_stats import combinations_of_2, gini
from query_encoder import encodeOps, getJoinProfile, encodeTypes
from explain_parser import ExplainParser as explain_parser

#####################   QUERY CLASS   #####################

def get_default_plan(sql,q_id,schema_name,conn_str,opt_plan_path):
    schema_name=schema_name.upper()

    exp_path,guideline,plan_cost=db2_explain(schema_name,sql,query_id=q_id,opt_plan_path=opt_plan_path,gen_exp_output=True,gen_guideline=True, return_cost=True,cmd_verbose=False, conn_str=conn_str)
    
    res = get_card_sel(conn_str)

    sel = {}
    card = {}
    for idx in range(len(res)):
        row = res.iloc[idx]
        sel['.'.join([schema_name,row.INPUT_OBJECT])] = row.SELECTIVITY
        card['.'.join([schema_name,row.INPUT_OBJECT])] = row.OUTPUT_CARD
    
    return exp_path, guideline, plan_cost, sel, card

def ext_sample_info(schema_name,encFileID,tables_dict,query_joins,local_preds,pred_cols,chai2matrixDict,conn_str):
    schema_name = schema_name.upper()

    samples_dir = './sample_data_{}_{}/'.format(schema_name.lower(),str(2000))
    multiPredj=False
    multiPred=0
    cyclic=0
    giniCoefDict = {}
    cardToColDict = {}
    table_datas = {}
    correlationsDict = {}
    pred_count = {}

    # load tables ordered by size and asign them to ids 
    id_tab = {}
    table_col_dict = load_db_schema(schema_name, conn_str)
    for idx, key in enumerate(table_col_dict.keys()):
        # tab = key.split('.')[1]
        id_tab[key] = idx

    # load db samples
    tables = list(tables_dict.keys())
    for table in tables:
        table = '.'.join([schema_name,tables_dict[table]])
        # Read csv files with "\" as escapechar and """ as quotechar. 
        tab_path = os.path.join(samples_dir,'{}_sample.csv'.format(table))
        table_datas[table] = pd.read_csv(tab_path, escapechar="\\", index_col=False, header=0)

    # create a dataframe containing query join preds
    jc_list = np.array(query_joins)[:,[2,5,6]]
    jpList = []
    for pred in jc_list:
        left_col, right_col, op = pred[0], pred[1], pred[2]
        left_tab = '.'.join(pred[0].split('.')[0:2])
        right_tab = '.'.join(pred[1].split('.')[0:2])
        jpList.append([left_tab,left_col,right_tab,right_col,op])
    jpArr = np.array(jpList)
    jpDF = pd.DataFrame(jpArr, columns = ["TL","CL","TR","CR","OP"])
    
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
        correlationsDict[tab] = {}
        if len(tabCols) == 1:
            correlationsDict[tab][tabCols[0]] = 1 
        else:
            colCombs = list(combinations_of_2(tabCols))
            for comb in colCombs:
                correlationsDict[tab][str(comb[0])+'-'+str(comb[1])] = chai2matrixDict[tab].loc[comb[0],comb[1]]

    # capture number of predicates per table
    pred_per_tab = predColDF.groupby(by='tabs').count()
    for table in id_tab.keys():
        if table in pred_per_tab.index:
            pred_count[table]=pred_per_tab.loc[table,'cols']
        else:
            pred_count[table]=0

    jColGroups = []
    for tr in jpDF.TR.unique():
        for tl in jpDF[jpDF.TR == tr].TL.unique():
            left_col_per_join = np.unique(jpDF[(jpDF.TL == tl) & (jpDF.TR == tr)].CL.values).tolist()
            right_col_per_join = np.unique(jpDF[(jpDF.TL == tl) & (jpDF.TR == tr)].CR.values).tolist()
            jColGroups.append([left_col_per_join,right_col_per_join])

    for jcgs in jColGroups:
        multiPredj = False
        for jcg in jcgs:
            jCols = [col.split(".")[2] for col in jcg]
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
            
            giniCoefDict[str(jcg)] = gini_coef
            cardToColDict[str(jcg)] = cardToCol

            if len(jcg) > 1:
                multiPredj=True
        if multiPredj:
            multiPred+=1

    pred_count["num_tab"] = len(list(tables_dict.keys()))
    pred_count["lpCount"] = len(local_preds)
    pred_count["multiPreds"] = multiPred
    pred_count["cyclic"] = cyclic
    pred_count["selfJoin"] = selfJoin
    pred_count["num_joins"] = eqJPred+rngJPred
    pred_count["eqJoinPreds"] = eqJPred
    pred_count["rangeJoinPreds"] = rngJPred
    
    return pred_count,correlationsDict,giniCoefDict,cardToColDict, id_tab, jpDF

def query_encoder(schema_name, max_tables, base_sels, base_cards, pred_count, giniCoefDict, cardToColDict, correlationsDict, joinIncsDict, joinTypesDict, joinFactorsDict, tables_dict, jpDF, id_tab):
   
    schema_name=schema_name.upper()
    
    pyG_encoding = {}

    AM = np.zeros((max_tables,max_tables,5)) # initialize Adjacency Matrix
    CR = np.zeros((max_tables,max_tables,1)) # initialize Card Ratio Matrix
    CC = np.zeros((max_tables,max_tables,2)) # initialize Card to Col Card Matrix
    IF = np.zeros((max_tables,max_tables,3)) # initialize Inclusion Factor Matrix
    JT = np.zeros((max_tables,max_tables,4)) # initialize Join Type Matrix
    JF = np.zeros((max_tables,max_tables,3)) # initialize Join Factor Matrix
    SK = np.zeros((max_tables,max_tables,2)) # initialize Skewness Factor
    NC = np.zeros((max_tables,6)) # initialize Node Coordinates Matrix

    jColGroups = {}
    for tr in jpDF.TR.unique():
        for tl in jpDF[jpDF.TR == tr].TL.unique():
            jColGroups[tl+'-'+tr] = [
                np.unique(jpDF[(jpDF.TL == tl) & (jpDF.TR == tr)].CL.values).tolist(),
                np.unique(jpDF[(jpDF.TL == tl) & (jpDF.TR == tr)].CR.values).tolist(),
                jpDF[(jpDF.TL == tl) & (jpDF.TR == tr)].OP.values.tolist()
                ]

    # get a list of tables referenced in the query
    table_list = ['.'.join([schema_name,str(tables_dict[key])]) for key in tables_dict]

    # if tables in the from clause of the query do not exist in base_cards and base_sels add them with with base_cards = 0 and base_sels = 0. This can happen when a table is eliminated in the rewrite phase.
    base_cards_keys = list(base_cards.keys())
    base_sels_keys = list(base_sels.keys())
    for table in table_list:
        if table not in base_cards_keys:
            base_cards[table] = 0
        if table not in base_sels_keys:
            base_sels[table] = 0

    for pred in jColGroups.keys():
        # print("pred --------->",pred)
        left_tab, right_tab = pred.split('-')
        left_col, right_col, predOp = jColGroups[pred][0], jColGroups[pred][1], jColGroups[pred][2]
        # Populate Adjacency Matrix
        AM[id_tab[left_tab],id_tab[right_tab]] = encodeOps(predOp)
        AM[id_tab[right_tab],id_tab[left_tab]] = encodeOps(predOp)

        # Populate Card Ratio Matrix
        card_ratio = max([base_cards[left_tab], base_cards[right_tab]])/(min([base_cards[left_tab], base_cards[right_tab]])+1)
        # card_ratio = base_cards.at[query_id, left_tab.split(".")[1]] / (base_cards.at[query_id, right_tab.split(".")[1]]+1)
        CR[id_tab[left_tab],id_tab[right_tab]] = card_ratio
        # CR[id_tab[right_tab],id_tab[left_tab]] = card_ratio
        CR[id_tab[right_tab],id_tab[left_tab]] = card_ratio

        # Populate card to col card ratio
        CC[id_tab[left_tab],id_tab[right_tab],0] = cardToColDict[str(left_col)]
        CC[id_tab[left_tab],id_tab[right_tab],1] = cardToColDict[str(right_col)]
        CC[id_tab[right_tab],id_tab[left_tab],0] = cardToColDict[str(left_col)]
        CC[id_tab[right_tab],id_tab[left_tab],1] = cardToColDict[str(right_col)]

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

        SK[id_tab[left_tab],id_tab[right_tab],0] = giniCoefDict[str(left_col)]
        SK[id_tab[left_tab],id_tab[right_tab],1] = giniCoefDict[str(right_col)]
        SK[id_tab[right_tab],id_tab[left_tab],0] = giniCoefDict[str(left_col)]
        SK[id_tab[right_tab],id_tab[left_tab],1] = giniCoefDict[str(right_col)]

    # Populate Node Coordinates Matrix
    for schema_tab in table_list:
        # tab = schema_tab.split('.')[1]
        NC[id_tab[schema_tab],0] = base_cards[schema_tab]
        NC[id_tab[schema_tab],1] = base_sels[schema_tab]
        NC[id_tab[schema_tab],2] = pred_count[schema_tab]
        # encode stats on pairwise correlations between predicate columns per table
        colCorDict = correlationsDict[schema_tab]
        corArr = np.array([colCorDict[comb] for comb in colCorDict.keys()])
        NC[id_tab[schema_tab],3] = corArr.min()
        NC[id_tab[schema_tab],4] = corArr.mean()
        NC[id_tab[schema_tab],5] = corArr.max()

    # organize the encoding in a format compatible with pytorch geometric
    edgeFeaturesMatrix = np.concatenate((AM,CR,CC,IF,JT,JF,SK), axis=2)
    nodeFeatures = NC.copy()

    # Construct edge indices
    AdjMat = np.max(AM, axis=2)
    edgeIndc=[]
    for i in range(AdjMat.shape[0]):
        for j in range(AdjMat.shape[1]):
            if AdjMat[i,j]>0:
                edgeIndc.append([i,j])
    edgeIndc = np.array(edgeIndc)
    edgeIndc = np.transpose(edgeIndc)


    # Construct edge features
    edgeFeat=[]
    for i in range(edgeIndc.shape[0]):
        if edgeIndc[i].sum() != 0:
            edgeFeat.append(edgeFeaturesMatrix[edgeIndc[i][0],edgeIndc[i][1],:])
    edgeFeat = np.array(edgeFeat)

    # construct Graph Features
    GF = [pred_count[key] for key in pred_count]
    GF = np.array(GF)

    # Populate the encodings and labels in the results dictionary
    pyG_encoding['nodeFeatures'] = (nodeFeatures)
    pyG_encoding['edgeIndc'] = (edgeIndc)
    pyG_encoding['edgeFeat'] = (edgeFeat)
    pyG_encoding['graphFeat'] = (GF)
    return pyG_encoding

class Query:
    def __init__(self, schema, sql, q_id, encFileID, conn_str_path = './conn_str', input_dir = './input/', opt_plan_path = './optimizer_plans/', internal_dir = './internal/', sample_size = 2000, verbose=False):
        self.schema = schema
        self.sql = sql
        self.q_id = q_id
        self.encFileID = encFileID
        self.conn_str_path = conn_str_path
        self.input_dir = input_dir
        self.opt_plan_path = opt_plan_path
        self.internal_dir = internal_dir
        self.sample_size = sample_size
        self.verbose = verbose
        self.plans = {}

    def parse(self):
        self.tables_dict,self.join_preds,self.local_preds,self.pred_cols= get_query_join_preds(self.schema,self.sql)
        if self.verbose:
            print (">>> parsed data stored in `tables_dict`, `join_preds`, `local_preds`, and `pred_cols` attributes.")
    
    def default_compile(self):

        with open(self.conn_str_path, "r") as conn_str_f:
            conn_str = conn_str_f.read()

        # get the default plan
        exp_path, guideline, plan_cost, self.sel, self.card= get_default_plan(self.sql,self.q_id,self.schema,conn_str,self.opt_plan_path)
        if self.verbose:
            print (">>> cardinaities and selectivities stored in `card` and `sel` attributes.")
        
        # parse the explain to get the mapping between aliases used in the guidelines and the table names
        parsed_exp = explain_parser(open(exp_path,'r').read()).parse()
        self.tab_alias_dict = parsed_exp['tab_alias_dic']
        
        # store the plan
        self.plans[0] = Plan(
            q_id=self.q_id,
            query=self.sql,
            hintset=[],hintset_id=0,
            explain_path=exp_path,guideline=guideline,
            cost=plan_cost,
            tab_alias_dict=self.tab_alias_dict,
            )
        
    def ext_sample_info(self,force_recollect=False):

        join_inc_path = os.path.join(self.internal_dir,'joinIncs_{}.json'.format(str(self.encFileID)))
        join_type_path = os.path.join(self.internal_dir,'joinTypes_{}.json'.format(str(self.encFileID)))
        join_factor_path = os.path.join(self.internal_dir,'joinFactors_{}.json'.format(str(self.encFileID)))
        chai2_matrix_path = os.path.join(self.internal_dir,'chai2matrixDict_{}.pickle'.format(str(self.encFileID)))

        with open(self.conn_str_path, "r") as conn_str_f:
            conn_str = conn_str_f.read()

        # db stats are only collected if they do not exist or if `force_recollect = True`
        if not (os.path.isfile(join_inc_path) and os.path.isfile(join_type_path) and os.path.isfile(join_factor_path) and os.path.isfile(chai2_matrix_path)) or force_recollect:

            from get_db_stats import get_db_stats
            get_db_stats(schema_name=self.schema, input_dir=self.input_dir, internal_dir=self.internal_dir, SAMPLE_SIZE=self.sample_size, encFileID=self.encFileID)

        # load all db stats
        with open(join_inc_path, 'r') as f:
            self.joinIncsDict = json.load(f)

        with open(join_type_path, 'r') as f:
            self.joinTypesDict = json.load(f)

        with open(join_factor_path, 'r') as f:
            self.joinFactorsDict = json.load(f)
        
        with open(chai2_matrix_path, 'rb') as f:
            self.chai2matrixDict = pickle.load(f)

        self.pred_count,self.correlationsDict,self.giniCoefDict,self.cardToColDict, self.id_tab, self.jpDF = ext_sample_info(self.schema,self.encFileID,self.tables_dict,self.join_preds,self.local_preds,self.pred_cols,self.chai2matrixDict, conn_str)
        self.max_tables = len(self.id_tab)
        
    def encode(self):
        self.parse()
        # only needed if the default plan does not exist
        # assumes the id for the default plan is 0
        if 0 not in self.plans.keys():
            self.default_compile()

        self.ext_sample_info()
        self.pyG_encoding = query_encoder(self.schema, self.max_tables, self.sel, self.card, self.pred_count, self.giniCoefDict, self.cardToColDict, self.correlationsDict, self.joinIncsDict, self.joinTypesDict, self.joinFactorsDict, self.tables_dict, self.jpDF, self.id_tab)

    def compile(self,hintset,hintset_id,gen_exp_output):
        with open(self.conn_str_path, "r") as conn_str_f:
            conn_str = conn_str_f.read()

        # compile the plan
        exp_path,guideline,plan_cost=db2_explain(self.schema,self.sql,self.q_id,hintset=hintset,hintset_id=hintset_id,opt_plan_path=self.opt_plan_path,gen_exp_output=gen_exp_output,gen_guideline=True, return_cost=True,cmd_verbose=self.verbose, conn_str=conn_str)
        
        # create a Plan object and assign it to the Query
        self.plans[hintset_id] = Plan(
            q_id=self.q_id,
            query=self.sql,
            hintset=hintset,hintset_id=hintset_id,
            explain_path=exp_path,guideline=guideline,
            cost=plan_cost,
            tab_alias_dict=self.tab_alias_dict,
            id_tab=self.id_tab
            )

    def execute(self,hintset,hintset_id,ibm_db_conn,timeout_thr=100, exec_verbose=False):
        
        if hintset_id not in self.plans.keys():
            self.compile(hintset,hintset_id,False)

        latency,errorMsg = db2_execute(self.sql,ibm_db_conn, self.plans[hintset_id].guideline,timeout_thr)
    
        self.plans[hintset_id].latency = latency
        self.plans[hintset_id].error_msg = errorMsg

        if latency >= timeout_thr:
            self.plans[hintset_id].timed_out = True
        else:
            self.plans[hintset_id].timed_out = False
            
        if self.verbose or exec_verbose:
            print("query {}, plan {}, latency: {}".format(str(self.q_id),str(hintset_id),str(latency)))
        
        return errorMsg
    
    def print(self):
        attributes = vars(self)
        print("Query Attributes:")
        print('\n'.join(['{}:{}'.format(attr, value) for attr, value in attributes.items()]))


#####################   PLAN CLASS   #####################

# prepare_trees funtion and related objects courtesy of Neo
from tcnn_util import featurizetree, xmltotree, prepare_trees,transformer,left_child,right_child
from util import find_between
import xml.etree.ElementTree as ET

def gltemplatetotree(string,tab_alias_dict):
    gl = find_between(string,'<OPTGUIDELINES>','</OPTGUIDELINES>')
    # print(gl)
    root = ET.fromstring(gl)
    featurizetree(root,tab_alias_dict)
    tree = xmltotree(root)
    return tree

def tabid2tab(guideline, tab_alias_dict):
    # Replaces table quantifier IDs in a guideline template with table names
    new_guideline = guideline
    for Q in tab_alias_dict.keys():
        new_guideline=new_guideline.replace("TABID=\'{}\'".format(Q),
                          "TABLE=\'{}\'".format(tab_alias_dict[Q]))
    return new_guideline
    
class Plan:
    def __init__(self,q_id=None,query=None,hintset_id=None, hintset=None, guideline=None,explain_path=None,tab_alias_dict=None,id_tab=None, cost=None, latency=None):
        self.q_id = q_id
        self.query = query
        self.hintset_id = hintset_id
        self.hintset = hintset
        self.guideline = guideline
        self.explain_path = explain_path
        self.tab_alias_dict = tab_alias_dict
        self.id_tab = id_tab
        self.cost = cost
        self.latency = latency

    def encode(self):
        # replace TABID's by table names
        guideline = tabid2tab(self.guideline, self.tab_alias_dict)
        tree = gltemplatetotree(guideline,self.id_tab)
        self.prep_tree=prepare_trees([tree],transformer, left_child, right_child, cuda=False)
    
    def print(self):
        attributes = vars(self)
        print("Plan Attributes:")
        print('\n'.join(['{}:{}'.format(attr, value) for attr, value in attributes.items()]))