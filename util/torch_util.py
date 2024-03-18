import numpy as np
import pandas as pd
from util.util import compute_org_m_s, extract_m_s_t, visual_eval, visual_eval_s2

def invers_transform_preds(preds,targetTransformer):
    mu_t = preds[:,0,:]
    var_t = preds[:,1,:]
    sigma_t = np.sqrt(var_t)
    mu_high_t = mu_t+sigma_t
    mu_low_t = mu_t-sigma_t
    mu = targetTransformer.inverse_transform(mu_t)
    mu_high = targetTransformer.inverse_transform(mu_high_t)
    mu_low = targetTransformer.inverse_transform(mu_low_t)
    return mu_low,mu,mu_high

def comparetopplansmulti(bm,bs2,p,verbose=False):
    N = bm.shape[0]
    import scipy.stats as st
    minS2 = bs2[bs2>0].min()
    bs2[bs2<=0]=minS2
    if verbose:
        # print("bestplanids",bestplanids[822])
        print("best_m",bm)
        print("best_s2",bs2)
        print("best_s",np.sqrt(bs2))
    M = np.zeros((N,p,p))
    S = np.zeros((N,p,p))
    Z = np.zeros((N,p,p))
    for n in range(N):
        print("query",n)
        for i in range(p):
            for j in range(p):
                M[n,i,j]=bm[n,i]-bm[n,j]
                if i!=j:
                    S[n,i,j]=np.sqrt(bs2[n,i]+bs2[n,j])
                else:
                    S[n,i,j]=1
    Z = np.multiply((0 - M),np.power(S,-1))
    R= 1-st.norm.cdf(Z)
    if verbose:
        print("M\n",M)
        print("S\n",S)
        print("Z\n",Z)
        print("R\n",R)

    return R.mean(axis=2)


# This evaluation function is compatible with the Pytorch model output 
# where there is only one label per sample
# Assumes pred is in the original space
def ETSubOpteval_s2(pred, elapsedTime, org_idx,
                    numSigma=1, targetTransformer = None, selectionRule=None, computeSubOptRisk = False,
                    slicer = None, num_plans = None):
    if slicer is not None:
        org_idx = org_idx[slicer]
    queryPlansDF = elapsedTime.loc[org_idx].copy()
    
    M_t, S2_t = extract_m_s_t(pred,targetTransformer)
    if slicer is not None:
        M_t, S2_t = M_t[slicer], S2_t[slicer]
    orgshape = M_t.shape
    # print('orgshape',orgshape)
    M, S_high, S_low = compute_org_m_s(M_t,S2_t,targetTransformer, 
                                        numSigma = numSigma,
                                        aggregate_s = False)
    _, S_high_1s, S_low_1s = compute_org_m_s(M_t,S2_t,targetTransformer, 
                                        numSigma = 1,
                                        aggregate_s = False) 

    M = M[org_idx.argsort()].reshape(-1)
    S_high = S_high[org_idx.argsort()].reshape(-1)
    S_low = S_low[org_idx.argsort()].reshape(-1)
    S_high_1s = S_high_1s[org_idx.argsort()].reshape(-1)
    S_low_1s = S_low_1s[org_idx.argsort()].reshape(-1)

    M_t = M_t[org_idx.argsort()].reshape(-1)
    S2_t = S2_t[org_idx.argsort()].reshape(-1)
    S_t = np.sqrt(S2_t)
    
    queryPlansDF["pred"] = M
    queryPlansDF["+s"] = S_high
    queryPlansDF["-s"] = S_low
    queryPlansDF["pred_high"] = M+S_high
    queryPlansDF["pred_low"] = M-S_low
    queryPlansDF["in_numsigma"] = False
    queryPlansDF.loc[((queryPlansDF.elapsed_time <= M+S_high)&
                     (queryPlansDF.elapsed_time >=  M-S_low)\
                     ),"in_numsigma"] = True    
    if targetTransformer is not None:
        queryPlansDF["norm_ET"] = targetTransformer.transform(queryPlansDF["elapsed_time"].values.reshape(-1,pred.shape[1])).reshape(-1)
    else:
        queryPlansDF["norm_ET"] = queryPlansDF["elapsed_time"]
    queryPlansDF["norm_m"] = M_t
    queryPlansDF["norm_s"] = S_t
    if computeSubOptRisk:
        queryPlansDF["SubOptRisk"]=np.nan
        for j in queryPlansDF.num_joins.unique():
            print(j,'joins')
            idx = (queryPlansDF.num_joins == j)
            num_plans = queryPlansDF[idx].guidelineID.nunique()
            R = comparetopplansmulti(M_t[idx].reshape(-1,num_plans), 
                                S2_t[idx].reshape(-1,num_plans),
                                num_plans,verbose=False).reshape(-1)
            queryPlansDF.loc[queryPlansDF.num_joins == j,"SubOptRisk"] = R
    queryPlansDF["calib_s"] = numSigma*S_t.reshape(-1)
    queryPlansDF["norm_m_high"] = M_t + numSigma*S_t
    queryPlansDF["norm_m_low"] = M_t - numSigma*S_t
    queryPlansDF["norm_in_2sigma"] = False
    queryPlansDF.loc[((queryPlansDF.norm_ET <= M_t + S_t)&
                     (queryPlansDF.norm_ET >= M_t - S_t)\
                     ),"norm_in_2sigma"] = True
    queryPlansDF["norm_in_4sigma"] = False
    queryPlansDF.loc[((queryPlansDF.norm_ET <= M_t + 2*S_t)&
                     (queryPlansDF.norm_ET >= M_t - 2*S_t)\
                     ),"norm_in_4sigma"] = True
    queryPlansDF["norm_in_6sigma"] = False
    queryPlansDF.loc[((queryPlansDF.norm_ET <= M_t + 3*S_t)&
                     (queryPlansDF.norm_ET >= M_t - 3*S_t)\
                     ),"norm_in_6sigma"] = True
    queryPlansDF["MLChoice"]=np.nan
    queryPlansDF["trueOpt"]=np.nan
    MLPlanPT = []
    trueOptPlanET = []
    query_ids = []
    for i in queryPlansDF.query_id.unique():
        query_ids.append(i)
        filtq = (queryPlansDF.query_id == i)
        dfSub = queryPlansDF[filtq]
        if selectionRule == 'conservative':
            MLPlanPT.append(dfSub.norm_m_high.min())
        elif selectionRule == 'suboptrisk':
            MLPlanPT.append(dfSub.SubOptRisk.min())
        else:
            MLPlanPT.append(dfSub.norm_m.min())
        
        trueOptPlanET.append(dfSub.elapsed_time.min())
        
        if selectionRule == 'conservative':
            queryPlansDF.loc[(filtq)&(queryPlansDF.norm_m_high == MLPlanPT[-1]), 'MLChoice'] = "optimal"
        elif selectionRule == 'suboptrisk':
            queryPlansDF.loc[(filtq)&(queryPlansDF.SubOptRisk == MLPlanPT[-1]), 'MLChoice'] = "optimal"
        else:
            queryPlansDF.loc[(filtq)&(queryPlansDF.norm_m == MLPlanPT[-1]), 'MLChoice'] = "optimal"
 
        queryPlansDF.loc[(queryPlansDF.query_id == i)&
         (queryPlansDF.elapsed_time == trueOptPlanET[-1]), 'trueOpt'] = "optimal"

    MLChoiceET = queryPlansDF[queryPlansDF.MLChoice == "optimal"].groupby('query_id',sort=False).first().elapsed_time.values
    optChoiceET = queryPlansDF[queryPlansDF.optChoice == "optimal"].elapsed_time.values
    trueOptimalET = np.array(trueOptPlanET)
    print("MLChoiceET",MLChoiceET.shape)
    print("trueOptimalET",trueOptimalET.shape)
    print("optChoiceET",optChoiceET.shape)
    MLplanETeval = pd.DataFrame([MLChoiceET,optChoiceET,trueOptimalET,
                                 MLChoiceET/trueOptimalET, 
                                 optChoiceET/trueOptimalET,
                                 MLChoiceET/optChoiceET]).T
    MLplanETeval.columns = ["MLET","OptimizerET","trueOptimalET","MLSubOpt","OptimizerSubOpt", "MLvsOptimizer"]
    MLplanETeval.index = query_ids
    return MLplanETeval, queryPlansDF

# A refactored version with way fewer number of parameters
def plotPerfOnQuery(query_idx, queryPlansDF, MLplanETeval, printSQL = True):
    
    msk = (queryPlansDF.query_id.isin([query_idx]))
    queryPlansDF_qry = queryPlansDF.loc[msk]
    mlplan = (queryPlansDF_qry.MLChoice == "optimal")
    optplan = (queryPlansDF_qry.optChoice == "optimal")
    bestplan = (queryPlansDF_qry.trueOpt == "optimal")
    
    act = queryPlansDF_qry.elapsed_time.values
    test = queryPlansDF_qry.pred.values

    testLable = " query " + str(query_idx)
    percentiles = [.01,.05,.1,.15,.25,.5,.75,.85,.9,.95,.97,.99]

    spPoints = {'ML Plan': mlplan.values, 'Optimizer Plan': optplan.values, 
               'Best Plan': bestplan.values}

    visual_eval(act,test,testLable,percentiles,spPoints)

    if printSQL:
        print("")
        print("ML's Choice:")
        print((queryPlansDF_qry.loc[mlplan]).query_sql.values[0])
        print("")
        print("Optimizer's Choice:")
        print((queryPlansDF_qry.loc[optplan]).query_sql.values[0])
        print("")
        print("True Optimal:")
        print((queryPlansDF_qry.loc[bestplan]).query_sql.values[0])


    print(MLplanETeval.loc[query_idx])

def compare_pipeline_results(res1,
                             res2,
                            label1,
                            label2):
    res1better = (res1.MLSubOpt < res2.MLSubOpt).mean()
    draw = (res1.MLSubOpt == res2.MLSubOpt).mean()
    print("pipeline {} is better than {} in {:.2%} percent of cases".format(label1,label2,res1better))    
    print("pipeline {} and pipeline {} were the same in {:.2%} percent of cases".format(label1,label2,draw))    
    
    totalres1imp = (res1.MLvsOptimizer < 1).sum()
    res2flippedres1imp = ((res1.MLvsOptimizer < 1)&(res2.MLvsOptimizer > 1)).sum()
    print("From {:.0f} cases improved by {} {:.0f} cases flipped to regressed by {}".format(totalres1imp,label1,res2flippedres1imp,label2))

    totalres1regr = (res1.MLvsOptimizer > 1).sum()
    res2flippedres1regr = ((res1.MLvsOptimizer > 1)&(res2.MLvsOptimizer < 1)).sum()
    print("From {:.0f} cases regressed by {} {:.0f} cases flipped to improved by {}".format(totalres1regr,label1,res2flippedres1regr,label2))

# A refactored version with way fewer number of parameters
def plotPerfOnQuery2(query_idx, queryPlansDF, MLplanETeval, printSQL = True,
                    numSigma = None,
                   targetTransformer=None, errorBars=None):
    
    msk = (queryPlansDF.query_id.isin([query_idx]))
    queryPlansDF_qry = queryPlansDF.loc[msk]
    mlplan = (queryPlansDF_qry.MLChoice == "optimal")
    optplan = (queryPlansDF_qry.optChoice == "optimal")
    bestplan = (queryPlansDF_qry.trueOpt == "optimal")
    
    act_t = queryPlansDF_qry.norm_ET.values.reshape(-1,1)
    M_t = queryPlansDF_qry.norm_m.values.reshape(-1,1)
    S2_t = np.square(queryPlansDF_qry.norm_s.values).reshape(-1,1)

    testLable = " query " + str(query_idx)
    percentiles = [.01,.05,.1,.15,.25,.5,.75,.85,.9,.95,.97,.99]

    spPoints = {'ML Plan': mlplan.values, 'Optimizer Plan': optplan.values, 
               'Best Plan': bestplan.values}

    visual_eval_s2(act_t,M_t,S2_t,testLable=testLable,
                percentiles=percentiles,spPoints=spPoints,
                   numSigma = numSigma,
                   targetTransformer=targetTransformer, errorBars=errorBars)

    if printSQL:
        print("")
        print("ML's Choice:")
        print((queryPlansDF_qry.loc[mlplan]).query_sql.values[0])
        print("")
        print("Optimizer's Choice:")
        print((queryPlansDF_qry.loc[optplan]).query_sql.values[0])
        print("")
        print("True Optimal:")
        print((queryPlansDF_qry.loc[bestplan]).query_sql.values[0])


    print(MLplanETeval.loc[query_idx])

def genLayerSizes(inputNumFeat = 12, firstLayerSize = 600, LastLayerSize = 30):
    if np.log2(firstLayerSize).is_integer():
        upperexp2 = firstLayerSize
    else:
        upperexp2 = (2**int(np.log2(firstLayerSize)))
    
    if np.log2(LastLayerSize).is_integer():
        lowerexp2 = LastLayerSize
    else:
        lowerexp2 = (2**int(np.log2(LastLayerSize)+1))
    # print('upperexp2',upperexp2)
    # print('lowerexp2',lowerexp2)
    layersizes = []
    layersizes.append(inputNumFeat)
    if firstLayerSize > upperexp2 and firstLayerSize != inputNumFeat:
        layersizes.append(firstLayerSize)
    nextvalue = upperexp2
    for i in range(100):
        if nextvalue >=lowerexp2 :
            if nextvalue != layersizes[-1]:
                layersizes.append(nextvalue)
            nextvalue=int(nextvalue/2)

        else:
            if LastLayerSize != lowerexp2:
                layersizes.append(LastLayerSize)
                break
    # print(layersizes)
    # layersizes = np.unique(np.array(layersizes))
    # print(layersizes)
    layers = []
    for idx, i in enumerate(layersizes):
        if idx <= len(layersizes)-2:
            layers.append([layersizes[idx], layersizes[idx+1]])
    # print(layers)
    return layers

# Enables variational inference using MC dropout
import torch.nn as nn
import pytorch_lightning as pl

class LitMCdropoutModel(pl.LightningModule):
    def __init__(self, model, mc_iteration = 10, dropout = 0.1):
        super().__init__()
        self.model = model
        self.mc_iteration = mc_iteration
    def predict_step(self, batch, batch_idx):
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Dropout):
                layer.train(True)
        preds = [self.model(batch).squeeze().tolist() for _ in range(self.mc_iteration)]
        return preds
