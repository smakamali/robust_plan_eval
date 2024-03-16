import os
import re
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_input_queries(input_dir):
    queries = []
    query_ids = []
    input_dir_enc = os.fsencode(input_dir)
    for file in os.listdir(input_dir_enc):
        filename = os.fsdecode(file)
        if filename.endswith(".sql"):
            query_ids.append(filename.split('.')[0])
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


from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

def lcm_model_eval(test_set= None, ypreds_tens_test_org= None, 
               dataset_label = None, model_label = None,
              percentiles = np.arange(0,1,0.1),load_from_disk=False, files_id = None,show_fig=True):
    import torch
    if load_from_disk:
        q_error_tensor,q_error_alt_tensor,num_joins_qe_tensor,ml_subOptTensor, db2_subOptTensor, num_joins_so_tensor = torch.load('./results/{}_{}_{}.pt'.format(model_label,dataset_label,files_id))
    else:
        from torch_scorers import subOpt2, q_error, q_error_alt
        ml_subOptList=[]
        db2_subOptList=[]
        q_error_list = []
        q_error_alt_list = []
        num_joins_list = []
        num_joins_qe_list = []
        for query_id in np.unique(test_set.query_id.numpy()):
            msk = (test_set.query_id == query_id)
            num_join = test_set.num_joins[msk][0]
            num_joins = test_set.num_joins[msk]
            targets = test_set.y[msk]
            preds = ypreds_tens_test_org[msk]
            opt_cost = test_set.opt_cost[msk]
            q_error_val = q_error(targets,preds)
            q_error_alt_val = q_error_alt(targets,preds)
            q_error_list.append(q_error_val)
            q_error_alt_list.append(q_error_alt_val)
            # opt_choice = (test_set.opt_choice[msk] == 1)
            ml_subOpt, db2_subopt = subOpt2(targets,preds,opt_cost)
            ml_subOptList.append(ml_subOpt)
            db2_subOptList.append(db2_subopt)
            num_joins_list.append(num_join)
            num_joins_qe_list.append(num_joins)
        ml_subOptTensor = torch.stack(ml_subOptList)
        db2_subOptTensor = torch.stack(db2_subOptList)
        num_joins_so_tensor = torch.stack(num_joins_list)
        q_error_tensor = torch.cat(q_error_list)
        q_error_alt_tensor = torch.cat(q_error_alt_list)
        num_joins_qe_tensor = torch.cat(num_joins_qe_list)
        torch.save((q_error_tensor,q_error_alt_tensor,num_joins_qe_tensor,ml_subOptTensor, db2_subOptTensor, num_joins_so_tensor),
                   './results/{}_{}_{}.pt'.format(model_label,dataset_label,files_id))
    
    # put results in dataframes
    q_error_num_joins = pd.DataFrame((q_error_tensor.numpy(), q_error_alt_tensor.numpy(),num_joins_qe_tensor.numpy())).T
    q_error_num_joins.columns = ['q_error','q_error_alt', 'num_joins']
    subopt_num_joins = pd.DataFrame((ml_subOptTensor.numpy(), db2_subOptTensor.numpy(),num_joins_so_tensor.numpy())).T
    subopt_num_joins.columns = ['ml_subopt','db2_subopt', 'num_joins']
    #plot results
    plot_data = [np.log10(q_error_num_joins[q_error_num_joins.num_joins == num].q_error_alt.values) for num in np.sort(q_error_num_joins.num_joins.unique())]
    plot_box_chart(plot_data, np.sort(q_error_num_joins.num_joins.unique()), 
               chart_title ='Cost estimation errors for {} using {}'.format(dataset_label, model_label),
               xlabel = 'num joins',
               yBounds = [-5,5], show_fig=show_fig)
    
    # summerize results
    from IPython.display import display
    from utility import plot_dist_line
    # q_error
    q_error_join_sum = q_error_num_joins[['q_error','num_joins']].groupby('num_joins').describe(percentiles=percentiles).T
    q_error_join_sum.index= q_error_join_sum.index.droplevel(0)
    q_error_sum = q_error_num_joins[['q_error']].describe(percentiles)
    q_error_sum = pd.concat([q_error_join_sum, q_error_sum], axis=1)
    q_error_sum = q_error_sum.rename(columns={'q_error':'overall q_error'})
    q_error_sum.to_csv('./results/q_error_sum_{}_{}_{}.csv'.format(model_label,dataset_label,files_id))
    display(q_error_sum)
    # subopt
    subopt_join_sum = subopt_num_joins[['ml_subopt','num_joins']].groupby('num_joins').describe(percentiles=percentiles).T
    db2_subopt_join_sum = subopt_num_joins[['db2_subopt','num_joins']].groupby('num_joins').describe(percentiles=percentiles).T
    subopt_join_sum.index= subopt_join_sum.index.droplevel(0)
    db2_subopt_join_sum.index= db2_subopt_join_sum.index.droplevel(0)
    subopt_sum = subopt_num_joins[['ml_subopt']].describe(percentiles)
    subopt_sum.columns = ['overall subopt']
    # display(subopt_sum)
    db2_subopt_sum = subopt_num_joins[['db2_subopt']].describe(percentiles)
    db2_subopt_sum.columns = ['overall subopt']

    dfs = {'ml_subopt':pd.concat([subopt_join_sum, subopt_sum], axis=1),
            'db2_subopt':pd.concat([db2_subopt_join_sum, db2_subopt_sum], axis=1)}
    subopt_sum = pd.concat(dfs.values(), axis=1, keys=dfs.keys())
    # subopt_sum = subopt_sum.rename(columns={'ml_subopt':'overall subopt'})
    # display(subopt_sum)
    subopt_sum.to_csv('./results/subopt_sum_{}_{}_{}.csv'.format(model_label,dataset_label,files_id))
    plot_dist_line(subopt_sum,'subopt','subopt for '+dataset_label+' using '+model_label,show_fig=show_fig)
    display(subopt_sum)

    return q_error_num_joins, subopt_sum

def plot_dist_line(data,metric_label,title,log_scale = True,
figsize=[10,6],plot_x_every=5,save_to=None,bbox_to_anchor=None,show_fig=True):
    index = [ind for ind in data.index if ind not in ["max","min","count","mean","std"]]
    indexPos = np.arange(len(index))
    fig = plt.figure(figsize=figsize)
    plt.plot(data.loc[index])
    plt.legend(data.columns, bbox_to_anchor=bbox_to_anchor)
    x = index
    y = [0]*len(index)
    plt.plot(x,y,color = 'tab:red')
    plt.xticks(ticks = indexPos[::plot_x_every], labels = index[::plot_x_every])
    if log_scale:
        plt.yscale('log')
    plt.grid(visible=True,which='both',axis='both')
    plt.title(title)
    plt.ylabel(metric_label)
    if save_to is not None:
        plt.savefig(save_to,bbox_inches='tight',dpi=300)
    if show_fig:
        plt.show()

def plot_box_chart(data, titles, chart_title = None, xlabel=None, yBounds=[-6, 6],show_fig=True):
    N=500
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title(chart_title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('log (prediction / actual)')

    # Now fill the boxes with desired colors
    box_colors = ['darkkhaki', 'royalblue']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='w', marker='*', markeredgecolor='k')

    # Set the axes ranges and axes labels
    # ax1.set_xlim(0.5, num_boxes + 0.5)
    # top = 40
    # bottom = -5
    ax1.set_ylim(yBounds)
    ax1.set_xticklabels(titles,
                        rotation=45, fontsize=12)

    pos = np.arange(num_boxes) + 1
    upper_labels = [str(np.round(s, 2)) for s in medians]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='small',
                 weight=weights[k], color=box_colors[k])
    if show_fig:
        plt.show()

def compareEstimates (act, test, base, testLable = "Test", baselineLabel = "Baseline"):
    percentiles=[.01,.05,.1,.15,.25,.5,.75,.85,.9,.95,.97,.99]
    meanQerr1,RMSE1,oneOM_1,error_df_1,_ = Evaluate_model(act, test)    
    meanQerr2,RMSE2,oneOM_2,error_df_2,_ = Evaluate_model(act, base)    
    plot_scatter(np.log10(error_df_1.actual),
             np.log10(error_df_1.prediction), 
             series=error_df_1[["series","series_sort"]], 
                          xlabel="Actual", ylabel="Predication", title="Compare log(cardinality) - "+testLable)
    plot_scatter(np.log10(error_df_2.actual),
             np.log10(error_df_2.prediction), 
             series=error_df_2[["series","series_sort"]], 
                          xlabel="Actual", ylabel="Predication", title="Compare log(cardinality) - "+baselineLabel)
    test_q_log = error_df_1.ratio_error_log
    base_q_log = error_df_2.ratio_error_log
    test_q_log.name = testLable
    base_q_log.name = baselineLabel
    pCorrTest, _ = pearsonr(test, act)
    pCorrBase, _ = pearsonr(base, act)
    
    qErrDF = pd.DataFrame([error_df_1.q_error,error_df_2.q_error]).T
    qErrDF.columns = ["Test","Baseline"]
    print("Qerror Distributions:")
    print(qErrDF.describe(percentiles))
    
    print("")
    print("RMSE for %s : %0.2f " %(testLable,RMSE1))
    print("RMSE for %s : %0.2f" %(baselineLabel,RMSE2))
    print("")
    print("within one order of magnitude for %s : %0.2f%% " %(testLable,oneOM_1*100))
    print("within one order of magnitude for %s : %0.2f%%" %(baselineLabel,oneOM_2*100))
    print("")
    print("Pearson's Correlation for %s : %0.5f " %(testLable, pCorrTest))
    print("Pearson's Correlation for %s : %0.5f " %(baselineLabel, pCorrBase))
    print("")
    compareLogQerr(test_q_log, base_q_log)

def Evaluate_model(y, predictions,percentiles=[.01,.05,.1,.15,.25,.5,.75,.85,.9,.95,.97,.99], unpacked=True):
    epsilon = min(np.min(predictions[np.nonzero(predictions)]), np.min(y[np.nonzero(y)]))
#     epsilon = 1
    predictions[predictions<=0]=epsilon
    y[y<=0]=epsilon
    
    MSE = mean_squared_error(y, predictions)
    RMSE = np.sqrt(mean_squared_error(y, predictions))
    oneOM = OrderMagnitude(y,predictions, 10)
    q_error_array = np.max(np.column_stack((y,predictions)), axis=1) / (np.min(np.column_stack((y,predictions)), axis=1))
    mean_q_error = q_error_array.mean()
    ratio_error_log = np.log10(predictions/y)
    error_df = pd.DataFrame([y, predictions, q_error_array, ratio_error_log], index=["actual", "prediction", "q_error", "ratio_error_log"]).T    
    error_df["ratio_error"]=error_df.prediction/error_df.actual
#     error_df["series"]=np.where(error_df.q_error>=10, ">oneOM", "<oneOM")
    error_df["series"]= error_df.apply(mark_OM, axis=1)
    error_df[["series","series_sort"]] = pd.DataFrame(error_df.series.tolist(), index = error_df.index)
    results = {"q_error_array":q_error_array,
               "mean_q_error":mean_q_error,
              "RMSE":RMSE,
              "oneOM":oneOM,
              "error_df":error_df,
              "error_df_sum":error_df.describe(percentiles)}
    if unpacked:
        return mean_q_error,RMSE,oneOM,error_df,error_df.describe(percentiles)
    else:
        return results

def OrderMagnitude (y_true, y_pred, tol):
    order_magnitude = np.max(np.column_stack((y_true,y_pred)), axis=1) / (np.min(np.column_stack((y_true,y_pred)), axis=1))
    OrderMagnitude = (order_magnitude <= tol).mean()
    return OrderMagnitude
def mark_OM(row):
    if row['q_error'] <= 2:
        return ['Y<2x', 1]
    elif row['q_error'] <= 5:
        return ['2x<Y<5x', 2] 
    elif row['q_error'] <= 10:
        return ['5x<Y<10x', 3] 
    else:
        return ['Y>10x', 4]
def plot_scatter(x,y,series=None,series_sort=None,xlabel=None, ylabel = None, title=None, 
                 x_max = None, y_max =None, x_min = None, y_min = None, 
                 plot_bounds = True, spPoints = None):
    colors=['tab:blue','tab:green','tab:orange', 'tab:red']
    if x_max is None:
        x_max = max(np.max(x),np.max(y))
    if y_max is None:
        y_max = max(np.max(x),np.max(y))
    if x_min is None:
        x_min = min(np.min(x),np.min(y))
    if y_min is None:
        y_min = min(np.min(x),np.min(y))
#     fig, ax = plt.subplots()
    plt.figure(figsize = (8,8))
    series_ref = series.drop_duplicates().sort_values(by="series_sort").series.values
    if series is not None:
        for num, name in enumerate(series_ref):
            mask = (series.series == name)
            plt.scatter(x[mask],y[mask],s=5, c=colors[num%4], label = name)
    else:
        plt.scatter(x,y,s=5)
    
    colors=['blue', 'orange', 'green', 'red']

    if spPoints is not None:
        for num, sp in enumerate(spPoints.keys()):
            plt.scatter(x[spPoints[sp]],y[spPoints[sp]],s=100,marker='x', 
                        c=colors[num%4], label = sp)
    
    x = np.linspace(x_min, x_max, 1000)
    y = x
    plt.plot(x,y,color = 'tab:blue')
#     y = 0*x
    y = x + 1
    if plot_bounds:
        plt.plot(x,y,linestyle='dashed', color = 'tab:orange')
    y = np.linspace(y_min, y_max, 1000)
#     x = 0*y
    y = x -1
    if plot_bounds:
        plt.plot(x,y,linestyle='dashed', color = 'tab:orange')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    plt.legend(loc='lower right')
    plt.show()

def compareLogQerr(qTest,qBase,percentiles=np.arange(0.01,1,0.01),
                  metricLabel = "q-error"):
    testLabel = qTest.name
    baselineLabel = qBase.name
    if 'elapsed_time' in metricLabel :
        eq = (qTest>qBase).mean()
        qDiff = qBase - qTest
    else:
        eq = (abs(qTest)>abs(qBase)).mean()
        qDiff = (abs(qBase) - abs(qTest))
    qDiff.rename("qDiff",inplace=True)
#     qDiffData = qDiff.describe(percentiles = percentiles+[eq])
    qDiffData = qDiff.describe(percentiles = percentiles)
    print ("%s method improves the %s for %.2f%% of the cases up to %.2f orders of magnitude compared with the %s." % (testLabel,metricLabel,(1-eq)*100,qDiffData["max"],baselineLabel))
    print ("%s method degrades the %s for %.2f%% of the cases up to %.2f orders of magnitude compared with the %s." % (testLabel,metricLabel,(eq)*100,abs(qDiffData["min"]),baselineLabel))
    qDiffData.drop(index=["count","mean","std"], inplace=True)
    positiveMsk = (qDiffData >= 0) 
    index = qDiffData.index.tolist()
    indexPos = np.arange(len(index))
    fig = plt.figure(figsize=[12,8])
    imp = plt.bar(x=indexPos[~positiveMsk], height=qDiffData[~positiveMsk], color = 'darksalmon')
    reg = plt.bar(x=indexPos[positiveMsk], height=qDiffData[positiveMsk], color = 'lightgreen')
    plt.legend([imp,reg],['Regressions','Improvements'])
    x = index
    y = [0]*len(index)
    plt.plot(x,y,color = 'tab:red')
    plt.xticks(ticks = indexPos[::5], labels = index[::5])
    if metricLabel == "q-error":
        plt.title("abs(log(baseline_prediction/actual)) - abs(log(test_prediction/actual))")
    else:
        plt.title(metricLabel)
    plt.show()
    
    qDF=pd.concat([qTest,qBase],axis=1)    
    data =qDF.describe(percentiles = percentiles)
    index = [ind for ind in data.index if ind not in ["count","mean","std"]]
    fig = plt.figure(figsize=[12,8])
    plt.plot(data.loc[index])
    plt.legend(data.columns)
    x = index
    y = [0]*len(index)
    plt.plot(x,y,color = 'tab:red')
    plt.xticks(ticks = indexPos[::5], labels = index[::5])
    if metricLabel == "q-error":
        plt.title("log(prediction/actual) %s vs. %s"%(qTest.name, qBase.name))
    else:
        plt.title("%s %s vs. %s"% (metricLabel, qTest.name, qBase.name))
    plt.show()

def visual_eval(act,test,testLable,percentiles, spPoints = None):
    from scipy.stats import pearsonr
    
    _,RMSE1,oneOM_1,error_df_1,err_dscribe = Evaluate_model(act, test, percentiles)    
    plot_scatter(np.log10(error_df_1.actual),
             np.log10(error_df_1.prediction), 
             series=error_df_1[["series","series_sort"]], 
                          xlabel="Actual", ylabel="Predication", 
                          title="Compare log(elapsed_time) - "+testLable, 
                          spPoints = spPoints)
    tmp = error_df_1.groupby(by=["series_sort","series"])[["series"]].count()\
                                            /error_df_1.shape[0]*100
    tmp.columns = [testLable]
    print(tmp)
    plot_scatter((error_df_1.actual),
             (error_df_1.prediction), 
             series=error_df_1[["series","series_sort"]], 
                          xlabel="Actual", ylabel="Predication", 
                          title="Compare (elapsed_time) - "+testLable, 
                          plot_bounds = False, spPoints = spPoints)
    print("Qerror Distributions:")
    print(err_dscribe.q_error)
    
    pCorrTest, _ = pearsonr(test, act)
    print("RMSE for %s : %0.2f " %(testLable,RMSE1))
    print("within one order of magnitude for %s : %0.2f%% " %(testLable,oneOM_1*100))
    print("Pearson's Correlation for %s : %0.5f " %(testLable, pCorrTest))
    

def plotPerfOnQuery(y, pred, org_idxs, query_idx, queryPlansDF, MLplanETeval, printSQL = True):
    
    idx = [query_idx,query_idx+1]
    act = y[idx[0]:idx[1]].reshape(-1)
    test = pred[idx[0]:idx[1]].reshape(-1)

    org_idx = org_idxs[idx[0]:idx[1]]

    testLable = "training hold-out query " + str(org_idx.tolist())
    percentiles = [.01,.05,.1,.15,.25,.5,.75,.85,.9,.95,.97,.99]

    msk = (queryPlansDF.query_id.isin(org_idx))
    queryPlansDF_qry = queryPlansDF.loc[msk]
    mlplan = (queryPlansDF_qry.MLChoice == "optimal")
    optplan = (queryPlansDF_qry.optChoice == "optimal")
    bestplan = (queryPlansDF_qry.trueOpt == "optimal")

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


    print(MLplanETeval.loc[org_idx])


def ETSubOpteval(pred, elapsedTime, org_idx):
    queryPlansDF = elapsedTime[elapsedTime.query_id.isin(org_idx)].copy()

    preds = pred[org_idx.argsort()]
    queryPlansDF["pred"] = preds.reshape(-1)

    queryPlansDF["MLChoice"]=np.nan
    queryPlansDF["trueOpt"]=np.nan
    MLPlanPT = []
    trueOptPlanET = []
    query_ids = []
    for i in queryPlansDF.query_id.unique():
        query_ids.append(i)
        MLPlanPT.append(queryPlansDF[queryPlansDF.query_id == i][["pred"]].min().values[0])
        trueOptPlanET.append(queryPlansDF[queryPlansDF.query_id == i][["elapsed_time"]].min().values[0])
        queryPlansDF.loc[(queryPlansDF.query_id == i)&
                           (queryPlansDF.pred == MLPlanPT[-1]), 'MLChoice'] = "optimal" 
        queryPlansDF.loc[(queryPlansDF.query_id == i)&
                           (queryPlansDF.elapsed_time == trueOptPlanET[-1]), 'trueOpt'] = "optimal"

    MLChoiceET = queryPlansDF[queryPlansDF.MLChoice == "optimal"].elapsed_time.values
    optChoiceET = queryPlansDF[queryPlansDF.optChoice == "optimal"].elapsed_time.values
    trueOptimalET = np.array(trueOptPlanET)
    MLplanETeval = pd.DataFrame([MLChoiceET,optChoiceET,trueOptimalET,
                                 MLChoiceET/trueOptimalET, 
                                 optChoiceET/trueOptimalET,
                                 MLChoiceET/optChoiceET]).T
    MLplanETeval.columns = ["MLET","OptimizerET","trueOptimalET","MLSubOpt","OptimizerSubOpt", "MLvsOptimizer"]
    MLplanETeval.index = query_ids
    return MLplanETeval, queryPlansDF

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    corr = np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))
    return np.diagonal(corr)

def extPerfStats(MLplanETeval, queryPlansDF= None, percentiles = [.01,.05,.1,.15,.25,.5,.75,.85,.9,.95,.97,.99]):
    results = {}
    improvedPortion = (MLplanETeval.MLvsOptimizer <= 1).mean()
    a = 1/MLplanETeval[MLplanETeval.MLvsOptimizer <= 1].MLvsOptimizer
    b = MLplanETeval[MLplanETeval.MLvsOptimizer <= 1].trueOptimalET
    meanImprovement = np.average(a, weights = b)
#     print("mean I:",a.mean())
#     print("weighted mean I:",meanImprovement)
    a = MLplanETeval[MLplanETeval.MLvsOptimizer > 1].MLvsOptimizer
    b = MLplanETeval[MLplanETeval.MLvsOptimizer > 1].trueOptimalET
    meanRegression = np.average(a, weights = b)
#     print("mean R:",a.mean())
#     print('weighted mean R:',meanRegression)
    results["improvedPortion"] = improvedPortion
    results["meanImprovement"] = meanImprovement
    results["meanRegression"] = meanRegression
    sumStats =  MLplanETeval.describe(percentiles)
    
    a = MLplanETeval.MLvsOptimizer
    b = MLplanETeval.trueOptimalET
    meanMLvDb2 = np.average(a, weights = b)
    # meanMLvDb2 = np.mean(a)
#     print("mean Net",sumStats["MLvsOptimizer"].loc['mean'])
#     print("weighted mean Net",meanMLvDb2)
    medianMLvDb2 = sumStats["MLvsOptimizer"].loc['50%']
    p99MLvDb2 = sumStats["MLvsOptimizer"].loc['99%']
    results["meanMLvDb2"]=meanMLvDb2
    results["medianMLvDb2"]=medianMLvDb2
    results["p99MLvDb2"]=p99MLvDb2
    
    a = MLplanETeval.MLSubOpt
    b = MLplanETeval.trueOptimalET
    meanSubOpt = np.average(a, weights = b)
    # meanSubOpt = np.mean(a)
    medianSubOpt = sumStats["MLSubOpt"].loc['50%']
    p99SubOpt = sumStats["MLSubOpt"].loc['99%']
    results["meanSubOpt"]=meanSubOpt
    results["medianSubOpt"]=medianSubOpt
    results["p99SubOpt"]=p99SubOpt
    
    a = MLplanETeval.OptimizerSubOpt
    b = MLplanETeval.trueOptimalET
    meanDb2SubOpt = np.average(a, weights = b)
    # meanDb2SubOpt = np.mean(a)
    medianDb2SubOpt = sumStats["OptimizerSubOpt"].loc['50%']
    p99Db2SubOpt = sumStats["OptimizerSubOpt"].loc['99%']
    results["meanDb2SubOpt"]=meanDb2SubOpt
    results["medianDb2SubOpt"]=medianDb2SubOpt
    results["p99Db2SubOpt"]=p99Db2SubOpt

    if queryPlansDF is not None:
        uniuqePlans=queryPlansDF.guidelineID.nunique()
        a = queryPlansDF.pred.values.reshape(-1,uniuqePlans)
        b = queryPlansDF.elapsed_time.values.reshape(-1,uniuqePlans)
        c = corr2_coeff(a,b)
        corr_sum = pd.Series(c).describe(percentiles)
        results["meanCorrelation"] = corr_sum.loc["mean"]
        results["medianCorrelation"] = corr_sum.loc["50%"]
        results["p1Correlation"] = corr_sum.loc["1%"]
        results["stdCorrelation"] = corr_sum.loc["std"]

        a = queryPlansDF.pred.values
        b = queryPlansDF.elapsed_time.values
        q_error_array = np.max(np.column_stack((a,b)), axis=1) / (np.min(np.column_stack((a,b)), axis=1))
        sumQerr = pd.Series(q_error_array).describe(percentiles)
        mean_qerror = sumQerr.loc["mean"]
        median_qerror = sumQerr.loc["50%"]
        p99_qerror  = sumQerr.loc["99%"]
        results['mean_qerror'] = mean_qerror
        results['median_qerror'] = median_qerror
        results['p99_qerror'] = p99_qerror
    
    results["sumStats"]=MLplanETeval.copy()
    return results

def compareSubOpt(MLplanETeval,percentiles=np.arange(0.01,1,0.01),
                  metricLabel = "q-error", 
                  testLabel = "ML Join Planner", 
                  baselineLabel = "Db2 Optimizer"):
    qTest = np.log10(MLplanETeval.MLSubOpt)
    qBase = np.log10(MLplanETeval.OptimizerSubOpt)
    
    perfStats = extPerfStats(MLplanETeval, percentiles = percentiles)
    improvedPortion = perfStats['improvedPortion']
    meanImprovement = perfStats['meanImprovement']
    meanRegression = perfStats['meanRegression']
    meanMLvDb2 = perfStats['meanMLvDb2']
    meanSubOpt = perfStats['meanSubOpt']
    medianSubOpt = perfStats['medianSubOpt']
    p99SubOpt = perfStats['p99SubOpt']
    meanDb2SubOpt = perfStats['meanDb2SubOpt']
    medianDb2SubOpt = perfStats['medianDb2SubOpt']
    p99Db2SubOpt = perfStats['p99Db2SubOpt']
    
    if 'elapsed_time' in metricLabel :
        eq = (qTest>qBase).mean()
        qDiff = qBase - qTest
    else:
        eq = (abs(qTest)>abs(qBase)).mean()
        qDiff = (abs(qBase) - abs(qTest))
    qDiff = -np.log10(MLplanETeval.MLvsOptimizer)
    qDiff.rename("qDiff",inplace=True)
#     qDiffData = qDiff.describe(percentiles = percentiles+[eq])
    qDiffData = qDiff.describe(percentiles = percentiles)
    print ("%s improves the %s for %.2f%% of the cases on average by %.2fx compared with the %s." % (testLabel,metricLabel,improvedPortion*100,meanImprovement,baselineLabel))
    print ("%s degrades the %s for %.2f%% of the cases on average by %.2fx compared with the %s." % (testLabel,metricLabel,(1-improvedPortion)*100,meanRegression,baselineLabel))
    
    if meanMLvDb2 > 1:
        print ("Overall %s degrades performance vs. %s by %.2fx." % (testLabel,baselineLabel,meanMLvDb2))
    else:
        print ("Overall %s improves performance vs. %s by %.2fx." % (testLabel,baselineLabel,1/meanMLvDb2))
    print('')
    print ("Mean SubOptimality of %s is %.2fx." % (testLabel,meanSubOpt))
    print ("Mean SubOptimality of %s is %.2fx." % (baselineLabel,meanDb2SubOpt))
    print('')
    print ("Median SubOptimality of %s is %.2fx." % (testLabel,medianSubOpt))
    print ("Median SubOptimality of %s is %.2fx." % (baselineLabel,medianDb2SubOpt))
    print('')
    print ("99th percentile of SubOptimality of %s is %.2fx." % (testLabel,p99SubOpt))
    print ("99th percentile of SubOptimality of %s is %.2fx." % (baselineLabel,p99Db2SubOpt))
    qDiffData.drop(index=["count","mean","std"], inplace=True)
    positiveMsk = (qDiffData >= 0) 
    index = qDiffData.index.tolist()
    indexPos = np.arange(len(index))
    fig = plt.figure(figsize=[12,8])
    imp = plt.bar(x=indexPos[~positiveMsk], height=qDiffData[~positiveMsk], color = 'darksalmon')
    reg = plt.bar(x=indexPos[positiveMsk], height=qDiffData[positiveMsk], color = 'lightgreen')
    plt.legend([imp,reg],['Regressions','Improvements'])
    x = index
    y = [0]*len(index)
    plt.plot(x,y,color = 'tab:red')
    plt.xticks(ticks = indexPos[::5], labels = index[::5])
    if metricLabel == "q-error":
        plt.title("abs(log(baseline_prediction/actual)) - abs(log(test_prediction/actual))")
    else:
        plt.title(metricLabel)
    plt.show()
    
    qDF=pd.concat([qTest,qBase],axis=1)    
    data =qDF.describe(percentiles = percentiles)
    index = [ind for ind in data.index if ind not in ["count","mean","std"]]
    fig = plt.figure(figsize=[12,8])
    plt.plot(data.loc[index])
    plt.legend(data.columns)
    x = index
    y = [0]*len(index)
    plt.plot(x,y,color = 'tab:red')
    plt.xticks(ticks = indexPos[::5], labels = index[::5])
    if metricLabel == "q-error":
        plt.title("log(prediction/actual) %s vs. %s"%(qTest.name, qBase.name))
    else:
        plt.title("%s %s vs. %s"% (metricLabel, qTest.name, qBase.name))
    plt.show()
def average(lst):
    nparr = np.array(lst)
    return np.mean(nparr)

def std(lst):
    nparr = np.array(lst)
    return np.std(nparr)

def visLC(df_tr,df_trho,label1="Train",label2="Test"):
    key1 = list(df_tr.keys())[-1]
    key2 = list(df_tr[list(df_tr.keys())[-1]].keys())[-1]
    for metric in list(df_tr[key1][key2].keys())[:-1]:
        
        temp = {}
        for key in df_tr.keys():
            temp[key] = []
            for fold in df_tr[key].keys():
                temp[key].append(df_tr[key][fold][metric])
        
        temp2 = {}
        for key in df_trho.keys():
            temp2[key] = []
            for fold in df_trho[key].keys():
                temp2[key].append(df_trho[key][fold][metric])
        
        clr1 = plt.cm.Blues(0.9)
        clr2 = plt.cm.Oranges(0.9)
        fig, ax1 = plt.subplots(1, 1, figsize = (8, 5))
        x = list(df_tr.keys())
        
        y_mean = np.array([average(temp[key]) for key in temp.keys()])
        y_std = np.array([std(temp[key]) for key in temp.keys()])
        y_low = y_mean - y_std
        y_high = y_mean + y_std
        ax1.plot(x,y_mean,label = label1, color = clr1)
        ax1.fill_between(x,y_low, y_high,alpha=0.3,edgecolor=clr1, facecolor=clr1)
        
        y_mean2 = np.array([average(temp2[key]) for key in temp2.keys()])
        y_std2 = np.array([std(temp2[key]) for key in temp2.keys()])
        y_low2 = y_mean2 - y_std2
        y_high2 = y_mean2 + y_std2
        ax1.plot(x,y_mean2,label = label2, color = clr2)
        ax1.fill_between(x,y_low2, y_high2,alpha=0.3,edgecolor=clr2, facecolor=clr2)
        
        metricLabel = metric
        ax1.set_title("%s"% (metricLabel))
        plt.legend()
        plt.show()

def multivisLC(dfs,labels=["Train","Test"]):
    clrs = [plt.cm.Blues(0.9), plt.cm.Oranges(0.9), plt.cm.Greens(0.9)] 
    # Assuming all dfs have the same set of metrics
    key1 = list(dfs[0].keys())[-1]
    key2 = list(dfs[0][list(dfs[0].keys())[-1]].keys())[-1]
    for metric in list(dfs[0][key1][key2].keys())[:-1]:
        plotdata = {}
        fig, ax1 = plt.subplots(1, 1, figsize = (8, 5))
        for idx,df_tr in enumerate(dfs):
            plotdata[idx] = {}
            key1 = list(df_tr.keys())[-1]
            key2 = list(df_tr[list(df_tr.keys())[-1]].keys())[-1]
            for key in df_tr.keys():
                plotdata[idx][key] = []
                for fold in df_tr[key].keys():
                    plotdata[idx][key].append(df_tr[key][fold][metric])
            temp = plotdata[idx]
            x = list(df_tr.keys())
            y_mean = np.array([average(temp[key]) for key in temp.keys()])
            y_std = np.array([std(temp[key]) for key in temp.keys()])
            y_low = y_mean - y_std
            y_high = y_mean + y_std
            ax1.plot(x,y_mean,label = labels[idx], color = clrs[idx])
            ax1.fill_between(x,y_low, y_high,alpha=0.3,edgecolor=clrs[idx], facecolor=clrs[idx])
        metricLabel = metric
        ax1.set_title("%s"% (metricLabel))
        plt.legend()
        plt.show()

def plotPerfOnQuery_s2(y, pred, org_idxs, query_idx, queryPlansDF, 
                       MLplanETeval, targetTransformer, 
                       numSigma=1, printSQL = True,errorBars=True):
    # print("query_idx",query_idx)
    idx = [query_idx]
    # print("idx",idx)
    org_idx = org_idxs[idx]
    # print("org_idx",org_idx)
    y_t = targetTransformer.transform(y)
    act_t = y_t[idx,:]
    # print("act_t",act_t)
    
    if pred.shape != y.shape:
        M_t, S2_t = extract_m_s_t(pred,targetTransformer)
        M_t = M_t[idx,:]
        S2_t = S2_t[idx,:]
    else:
        M_t = pred[idx]
        S2_t = None


    testLable = "query " + str(org_idx)
    percentiles = [.01,.05,.1,.15,.25,.5,.75,.85,.9,.95,.97,.99]

    msk = (queryPlansDF.query_id.isin(org_idx.tolist()))
    queryPlansDF_qry = queryPlansDF.loc[msk]
    mlplan = (queryPlansDF_qry.MLChoice == "optimal")
    optplan = (queryPlansDF_qry.optChoice == "optimal")
    bestplan = (queryPlansDF_qry.trueOpt == "optimal")

    spPoints = {'ML Plan': mlplan.values, 'Optimizer Plan': optplan.values, 
               'Best Plan': bestplan.values}

    visual_eval_s2(act_t,M_t,S2_t,testLable=testLable,
                percentiles=percentiles,spPoints=spPoints,
                   numSigma = numSigma,
                   targetTransformer=targetTransformer, errorBars=errorBars)

    if printSQL:
        print("")
        print("ML's Choice:")
        print("Guieline ID:",(queryPlansDF_qry.loc[mlplan]).guidelineID.values[0])
        print((queryPlansDF_qry.loc[mlplan]).query_sql.values[0])
        print("")
        print("Optimizer's Choice:")
        print("Guieline ID:",(queryPlansDF_qry.loc[optplan]).guidelineID.values[0])
        print((queryPlansDF_qry.loc[optplan]).query_sql.values[0])
        print("")
        print("True Optimal:")
        print("Guieline ID:",(queryPlansDF_qry.loc[bestplan]).guidelineID.values[0])
        print((queryPlansDF_qry.loc[bestplan]).query_sql.values[0])


    print(MLplanETeval.loc[org_idx])

def visual_eval_s2(act_t,M_t,S2_t=None,testLable=None,
                  percentiles=None, spPoints = None,
                   numSigma=1, targetTransformer = None, errorBars = True):
    from scipy.stats import pearsonr

    act = targetTransformer.inverse_transform(act_t).reshape(-1)

    M, S_high, S_low = compute_org_m_s(M_t,S2_t,numSigma =numSigma,
                                       aggregate_s = False,
                               targetTransformer=targetTransformer)
    M, S_high, S_low = M.reshape(-1), S_high.reshape(-1), S_low.reshape(-1)
    
    _,RMSE1,oneOM_1,error_df_1,err_dscribe = Evaluate_model(act, M, percentiles)     
    S_t = np.sqrt(S2_t)

    plot_scatter_s2(act_t.reshape(-1), M_t.reshape(-1), 
                    S_t.reshape(-1)*numSigma,
                 series=error_df_1[["series","series_sort"]], 
                          xlabel="Actual", ylabel="Predication", 
                          title="Compare log(elapsed_time) - "+testLable, 
                    x_max = 1, y_max =1, x_min = 0, y_min = 0, 
                          spPoints = spPoints, errorBars=errorBars)
    
    tmp = error_df_1.groupby(by=["series_sort","series"])[["series"]].count()\
                                            /error_df_1.shape[0]*100
    tmp.columns = [testLable]
    print(tmp)
    plot_scatter_s2(act, M, [S_high, S_low], 
                   series=error_df_1[["series","series_sort"]], 
                          xlabel="Actual", ylabel="Predication", 
                          title="Compare (elapsed_time) - "+testLable, 
                          plot_bounds = False, spPoints = spPoints, errorBars=errorBars)
    print("Qerror Distributions:")
    print(err_dscribe.q_error)
    
    pCorrTest, _ = pearsonr(M, act)
    print("RMSE for %s : %0.2f " %(testLable,RMSE1))
    print("within one order of magnitude for %s : %0.2f%% " %(testLable,oneOM_1*100))
    print("Pearson's Correlation for %s : %0.5f " %(testLable, pCorrTest))
    
def plot_scatter_s2(x,y,s=None,series=None,series_sort=None,xlabel=None, ylabel = None, title=None, 
                 x_max = None, y_max =None, x_min = None, y_min = None, 
                 plot_bounds = True, spPoints = None, errorBars = True):
    
    colors=['tab:blue','tab:green','tab:orange', 'tab:red']
    
    if x_max is None:
        x_max = max(np.max(x),np.max(y))
    if y_max is None:
        y_max = max(np.max(x),np.max(y))
    if x_min is None:
        x_min = min(np.min(x),np.min(y))
    if y_min is None:
        y_min = min(np.min(x),np.min(y))

    if isinstance(s, list):
        S_center = np.mean(np.stack([y + s[0],y - s[1]]),axis=0)
        S_mean = np.mean(np.stack([s[0],s[1]]),axis=0)

    plt.figure(figsize = (8,8))
    series_ref = series.drop_duplicates().sort_values(by="series_sort").series.values
    if series is not None:
        for num, name in enumerate(series_ref):
            mask = (series.series == name)
            if isinstance(s, list):
                plt.scatter(x[mask],y[mask],s=30, marker='x', c=colors[num%4], label = name)
                if errorBars:
                    plt.errorbar(x[mask],S_center[mask], yerr=S_mean[mask],c=colors[num%4],fmt='none')
            else:
                plt.scatter(x[mask],y[mask],s=30, marker='x', c=colors[num%4], label = name)
                if errorBars:
                    plt.errorbar(x[mask],y[mask], yerr=s[mask],c=colors[num%4],fmt='none')
            
    else:
        plt.scatter(x,y,s=5)
        plt.errorbar(x,y,yerr=s)
    
    colors=['purple', 'magenta', 'black']

    if spPoints is not None:
        for num, sp in enumerate(spPoints.keys()):
            if isinstance(s, list):
                plt.scatter(x[spPoints[sp]],y[spPoints[sp]],s=500,marker='x', 
                            c=colors[num%3], label = sp)
                if errorBars:
                    plt.errorbar(x[spPoints[sp]],S_center[spPoints[sp]], 
                            yerr=S_mean[spPoints[sp]],c=colors[num%3],fmt='none')
            else:
                plt.scatter(x[spPoints[sp]],y[spPoints[sp]],s=500,marker='x', 
                            c=colors[num%3], label = sp)
                if errorBars:
                    plt.errorbar(x[spPoints[sp]],y[spPoints[sp]], 
                            yerr=s[spPoints[sp]],c=colors[num%3],fmt='none')
    
    x = np.linspace(x_min, x_max, 1000)
    y = x
    plt.plot(x,y,color = 'tab:blue')
#     y = 0*x
    y = x + 1
    if plot_bounds:
        plt.plot(x,y,linestyle='dashed', color = 'tab:orange')
    y = np.linspace(y_min, y_max, 1000)
#     x = 0*y
    y = x -1
    if plot_bounds:
        plt.plot(x,y,linestyle='dashed', color = 'tab:orange')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    plt.legend(loc='lower right')
    plt.show()    

# This functions takes the prediction from aleatoric model as input
# and transforms it to the normal space using the targetTransformer
# then extracts mean and variance in the normal space
def extract_m_s_t(pred, targetTransformer = None):
    if targetTransformer is not None:
        pred_t = targetTransformer.transform(pred)
    else:
        pred_t = pred
    pred_rsh = pred_t.reshape(-1,2,pred_t.shape[1])
    M_t = pred_rsh[:,0,:]
    S2_t = pred_rsh[:,1,:]
    return M_t, S2_t

# The inverse of extract_m_s_t, 
# Gives the composed array in the original space
# Note that the ouput variance is not meaningful and 
# needs further processing 
# the output is compatible with ETSubOpteval_* functions
def compose_m_s2_t(m,s2,targetTransformer=None):
    stacked_m_st = np.stack([m,s2],axis=1)
    stacked_m_st=stacked_m_st.reshape(-1,m.shape[1])
    if targetTransformer is not None:
        stacked_m_st = targetTransformer.inverse_transform(stacked_m_st)
    return stacked_m_st

# This function takes the mean and variance in the normal space as input
# It then computes the standard deviation in the normal space
# Using the standard deviation, it computes mu+sigma and mu-sigma 
# as two bounds are then inverse transformed to the orginal space 
# The resulted bounds represent the interval in the orginal space that
# corresponds to mu+/-sigma in the normal space
def compute_org_m_s(M_t,S2_t,targetTransformer, numSigma =1, 
                    aggregate_s = True):
    S2_t[(S2_t<0)] = 0
    S_t = np.sqrt(S2_t)
    M_t_high = M_t + numSigma*S_t
    M_t_low = M_t - numSigma*S_t
    if targetTransformer is not None:
        M_high = targetTransformer.inverse_transform(M_t_high)
        M_low = targetTransformer.inverse_transform(M_t_low)
        M = targetTransformer.inverse_transform(M_t)
    else:
        M_high = M_t_high
        M_low = M_t_low
        M = M_t
    S_high = (M_high - M)
    S_low = (M - M_low)
    if aggregate_s:
        S_hat = np.mean(np.stack([S_high,S_low]),axis=0).reshape(S2_t.shape)
        return M, S_hat
    else:
        return M, S_high, S_low

# def calibrate_s(pred, elapsedTime, org_idx, 
#                     targetTransformer = None, conservative=False,
#                     useTransPred = False):
#     # queryPlansDF = elapsedTime[elapsedTime.query_id.isin(org_idx)].copy()
#     M_t, S2_t = extract_m_s_t(pred,targetTransformer)
    
#     M, S_high, S_low = compute_org_m_s(M_t,S2_t,targetTransformer, 
#                                            numSigma = numSigma,
#                                            aggregate_s = False) 
#     return None
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

# Extracts model and data uncertainty from an array of multiple predictions
def extract_m_s2a_s2e_t(pred_arr, y, targetTransformer=None):
    m_t_arr = []
    s2_t_arr = []
    for pred in pred_arr:
        m_t,s2_t = extract_m_s_t(pred, 
                                 targetTransformer=targetTransformer)
        m_t_arr.append(m_t)
        s2_t_arr.append(s2_t)
    m_t_arr = np.array(m_t_arr)
    s2_t_arr = np.array(s2_t_arr)
    mean_m_t = np.mean(m_t_arr,axis=0)
    s2a_t = (np.mean((s2_t_arr + np.square(m_t_arr)),axis = 0) - np.square(mean_m_t))
    s2e_t = np.var(m_t_arr,axis=0)
    return mean_m_t, s2a_t, s2e_t

def ETSubOpteval_s2(pred, elapsedTime, org_idx, 
                    numSigma=1, targetTransformer = None, selectionRule=None, computeSubOptRisk = False,
                    slicer = None):
    if slicer is not None:
        org_idx = org_idx[slicer]
    queryPlansDF = elapsedTime[elapsedTime.query_id.isin(org_idx)].copy()
    M_t, S2_t = extract_m_s_t(pred,targetTransformer)
    if slicer is not None:
        M_t, S2_t = M_t[slicer], S2_t[slicer]
    orgshape = M_t.shape
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
    queryPlansDF["norm_ET"] = targetTransformer.transform(queryPlansDF["elapsed_time"].values.reshape(-1,pred.shape[1])).reshape(-1)
    queryPlansDF["norm_m"] = M_t
    queryPlansDF["norm_s"] = S_t
    if computeSubOptRisk:
        R = comparetopplansmulti(M_t.reshape(orgshape), 
                                S2_t.reshape(orgshape),
                                orgshape[1],verbose=False).reshape(-1)
        queryPlansDF["SubOptRisk"] = R
    # queryPlansDF["calib_s"] = numSigma*S_t.reshape(-1)
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
        
        if selectionRule == 'conservative':
            MLPlanPT.append(queryPlansDF[queryPlansDF.query_id == i][["norm_m_high"]].min().values[0])
        elif selectionRule == 'suboptrisk':
            MLPlanPT.append(queryPlansDF[queryPlansDF.query_id == i][["SubOptRisk"]].min().values[0])
        else:
            MLPlanPT.append(queryPlansDF[queryPlansDF.query_id == i][["norm_m"]].min().values[0])
        
        trueOptPlanET.append(queryPlansDF[queryPlansDF.query_id == i][["elapsed_time"]].min().values[0])
        
        if selectionRule == 'conservative':
            queryPlansDF.loc[(queryPlansDF.query_id == i)&\
                    (queryPlansDF.norm_m_high == MLPlanPT[-1]), 'MLChoice'] = "optimal"
        elif selectionRule == 'suboptrisk':
            queryPlansDF.loc[(queryPlansDF.query_id == i)&\
                    (queryPlansDF.SubOptRisk == MLPlanPT[-1]), 'MLChoice'] = "optimal"
        else:
            queryPlansDF.loc[(queryPlansDF.query_id == i)&\
                    (queryPlansDF.norm_m == MLPlanPT[-1]), 'MLChoice'] = "optimal"
 
        queryPlansDF.loc[(queryPlansDF.query_id == i)&
         (queryPlansDF.elapsed_time == trueOptPlanET[-1]), 'trueOpt'] = "optimal"

    MLChoiceET = queryPlansDF[queryPlansDF.MLChoice == "optimal"].elapsed_time.values
    optChoiceET = queryPlansDF[queryPlansDF.optChoice == "optimal"].elapsed_time.values
    trueOptimalET = np.array(trueOptPlanET)
    MLplanETeval = pd.DataFrame([MLChoiceET,optChoiceET,trueOptimalET,
                                 MLChoiceET/trueOptimalET, 
                                 optChoiceET/trueOptimalET,
                                 MLChoiceET/optChoiceET]).T
    MLplanETeval.columns = ["MLET","OptimizerET","trueOptimalET","MLSubOpt","OptimizerSubOpt", "MLvsOptimizer"]
    MLplanETeval.index = query_ids
    return MLplanETeval, queryPlansDF

# This function has an option to compute soboptimality risk using
# both data and model uncertainties
def ETSubOpteval_s2ae(M_t, S2_t, S2_t_e, elapsedTime, org_idx, 
                    numSigma=1, targetTransformer = None, selectionRule=None, computeSubOptRisk = False,
                    slicer = None):
    if slicer is not None:
        org_idx = org_idx[slicer]
    queryPlansDF = elapsedTime[elapsedTime.query_id.isin(org_idx)].copy()
        
    # M_t, S2_t = extract_m_s_t(pred,targetTransformer)
    if slicer is not None:
        M_t, S2_t, S2_t_e = M_t[slicer], S2_t[slicer], S2_t_e[slicer]
    orgshape = M_t.shape
    M, S_high, S_low = compute_org_m_s(M_t,S2_t,targetTransformer, 
                                        numSigma = numSigma,
                                        aggregate_s = False)
    
    # _, S_high_1s, S_low_1s = compute_org_m_s(M_t,S2_t,targetTransformer, 
    #                                     numSigma = 1,
    #                                     aggregate_s = False) 

    M = M[org_idx.argsort()].reshape(-1)
    S_high = S_high[org_idx.argsort()].reshape(-1)
    S_low = S_low[org_idx.argsort()].reshape(-1)
    # S_high_1s = S_high_1s[org_idx.argsort()].reshape(-1)
    # S_low_1s = S_low_1s[org_idx.argsort()].reshape(-1)

    M_t = M_t[org_idx.argsort()].reshape(-1)
    S2_t = S2_t[org_idx.argsort()].reshape(-1)
    S2_t_e = S2_t_e[org_idx.argsort()].reshape(-1)
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
    print("orgshape.shape",orgshape[1])
    queryPlansDF["norm_ET"] = targetTransformer.transform(queryPlansDF["elapsed_time"].values.reshape(-1,orgshape[1])).reshape(-1)
    queryPlansDF["norm_m"] = M_t
    queryPlansDF["norm_s"] = S_t
    if computeSubOptRisk:
        if selectionRule == 'suboptriskA' or selectionRule == 'suboptrisk':
            Ra = comparetopplansmulti(M_t.reshape(orgshape), 
                                    S2_t.reshape(orgshape),
                                    orgshape[1],verbose=False).reshape(-1)
            queryPlansDF["suboptriskA"] = Ra
            
        if selectionRule == 'suboptriskE' or selectionRule == 'suboptrisk':
            Re = comparetopplansmulti(M_t.reshape(orgshape), 
                                S2_t_e.reshape(orgshape),
                                orgshape[1],verbose=False).reshape(-1)
            queryPlansDF["suboptriskE"] = Re
            
        if selectionRule == 'suboptrisk':
            queryPlansDF["suboptrisk"] = Ra + Re

        if selectionRule == 'suboptriskT':
            Rt = comparetopplansmulti(M_t.reshape(orgshape), 
                                (S2_t+S2_t_e).reshape(orgshape),
                                orgshape[1],verbose=False).reshape(-1)
            queryPlansDF["suboptriskT"] = Rt

    # queryPlansDF["calib_s"] = numSigma*S_t.reshape(-1)
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
        
        if selectionRule == 'conservative':
            MLPlanPT.append(queryPlansDF[queryPlansDF.query_id == i][["norm_m_high"]].min().values[0])
        elif 'suboptrisk' in selectionRule:
            MLPlanPT.append(queryPlansDF[queryPlansDF.query_id == i][[selectionRule]].min().values[0])
        else:
            MLPlanPT.append(queryPlansDF[queryPlansDF.query_id == i][["norm_m"]].min().values[0])
        
        trueOptPlanET.append(queryPlansDF[queryPlansDF.query_id == i][["elapsed_time"]].min().values[0])
        
        if selectionRule == 'conservative':
            queryPlansDF.loc[(queryPlansDF.query_id == i)&\
                    (queryPlansDF.norm_m_high == MLPlanPT[-1]), 'MLChoice'] = "optimal"
        elif 'suboptrisk' in selectionRule:
            queryPlansDF.loc[(queryPlansDF.query_id == i)&\
                    (queryPlansDF[selectionRule] == MLPlanPT[-1]), 'MLChoice'] = "optimal"
        else:
            queryPlansDF.loc[(queryPlansDF.query_id == i)&\
                    (queryPlansDF.norm_m == MLPlanPT[-1]), 'MLChoice'] = "optimal"
 
        queryPlansDF.loc[(queryPlansDF.query_id == i)&
         (queryPlansDF.elapsed_time == trueOptPlanET[-1]), 'trueOpt'] = "optimal"

    MLChoiceET = queryPlansDF[queryPlansDF.MLChoice == "optimal"].elapsed_time.values
    optChoiceET = queryPlansDF[queryPlansDF.optChoice == "optimal"].elapsed_time.values
    trueOptimalET = np.array(trueOptPlanET)
    MLplanETeval = pd.DataFrame([MLChoiceET,optChoiceET,trueOptimalET,
                                 MLChoiceET/trueOptimalET, 
                                 optChoiceET/trueOptimalET,
                                 MLChoiceET/optChoiceET]).T
    MLplanETeval.columns = ["MLET","OptimizerET","trueOptimalET","MLSubOpt","OptimizerSubOpt", "MLvsOptimizer"]
    MLplanETeval.index = query_ids
    return MLplanETeval, queryPlansDF

def evalCalib(queryPlansDF,title=''):
    import scipy.stats as st
    CLs= []
    withinIntervals = []
    for alpha in np.arange(0,1.01,0.01):
        p = 1- alpha/2
        z = st.norm.ppf(p)
        highBound = queryPlansDF.norm_m + z*queryPlansDF.norm_s
        lowBound = queryPlansDF.norm_m - z*queryPlansDF.norm_s
        withinBounds = ((queryPlansDF.norm_ET <= highBound) & (queryPlansDF.norm_ET >= lowBound)).sum()
        CLs.append(1-alpha)
        withinIntervals.append(withinBounds/queryPlansDF.shape[0])
    CLs = np.array(CLs)
    withinIntervals = np.array(withinIntervals)
    a = CLs
    b = withinIntervals
    mask = np.isfinite(b)
    plt.figure(figsize = (8,6))
    plt.plot(a[mask],b[mask],color = 'tab:orange')
    x = np.linspace(a.min(), a.max(), 1000)
    y = x
    plt.plot(x,y,color = 'tab:blue')
    plt.title("Calibration plot for "+title)
    plt.xlabel("Expected confidence interval")
    plt.ylabel("Observed confidence interval")
    plt.show()

def plotxvsy(x, y, xl, yl,scale='linear',lims=None):
    a = x.reshape(-1,1)
    b = y.reshape(-1,1)
    plt.figure(figsize = (6,6))
    plt.xlabel(xl)
    plt.ylabel(yl)
    if lims is not None:
        plt.xlim(lims)
        plt.ylim(lims)
    plt.xscale(scale)
    plt.yscale(scale)
    plt.title(xl+" vs. "+yl)
    plt.scatter(a,b, s=5)
    plt.show()

def replaceZerosByMin(arr):
    minval = np.min(arr[np.nonzero(arr)])
    arr[arr==0]=minval
    return arr

def plothist(a, labels=None,title='',bins=50, logscale = False):
    def plot_hist(arr, bins=bins,logscale=logscale):
        if logscale:
            arr = replaceZerosByMin(arr)
            plt.subplot(211)
            _, bins, _ = plt.hist(arr,bins=bins)
            if labels is not None:
                plt.legend(labels)
            plt.title(title)
            
            plt.subplot(212)
            logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
            plt.hist(arr, bins=logbins)
            plt.xscale('log')
            if labels is not None:
                plt.legend(labels)
            plt.show()

        else:
            plt.hist(arr,bins=bins)
            if labels is not None:
                plt.legend(labels)
            plt.title(title)
            plt.show()
        
    
    if isinstance(a[0],list):
        for arr in a:
            arr = np.array(arr)
            plot_hist(arr,bins=bins,logscale=logscale)
    else:
        arr = np.array(a)
        plot_hist(arr,bins=bins,logscale=logscale)
        
def plot2dhist(x,y, xl='', yl='', scale='linear'):
    x = x.reshape(-1)
    y = y.reshape(-1)
    # Creating bins
    x_min = np.min(x)
    x_max = np.max(x)
    
    y_min = np.min(y)
    y_max = np.max(y)
    
    x_bins = np.linspace(x_min, x_max, 100)
    y_bins = np.linspace(y_min, y_max, 100)
    
    fig, ax = plt.subplots(figsize =(10, 7))
    # Creating plot
    plt.hist2d(x, y, bins =[x_bins, y_bins], cmap = plt.cm.nipy_spectral)
    plt.title(xl+" vs. "+yl)
    plt.xscale(scale)
    plt.yscale(scale)
    # Adding color bar
    plt.colorbar()
    
    ax.set_xlabel(xl) 
    ax.set_ylabel(yl) 
    
    # show plot
    plt.tight_layout() 
    plt.show()    

def comparetopplans(m,s2,n,verbose=False):
    # m = m[]
    import scipy.stats as st
    bestplanids = m.argsort()[:n]
    bm = m[bestplanids]
    bs2 = s2[bestplanids]
    if verbose:
        print("bestplanids",bestplanids)
        print("best_m",bm)
        print("best_s2",bs2)
        print("best_s",np.sqrt(bs2))
    M = np.zeros((n,n))
    S = np.zeros((n,n))
    Z = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            M[i,j]=bm[i]-bm[j]
            if i!=j:
                S[i,j]=np.sqrt(bs2[i]+bs2[j])
            else:
                S[i,j]=1
    Z = np.multiply((0 - M),np.power(S,-1))
    R= 1-st.norm.cdf(Z)
    if verbose:
        print("M\n",M)
        print("S\n",S)
        print("Z\n",Z)
        print("R\n",R)

    return R,bestplanids

