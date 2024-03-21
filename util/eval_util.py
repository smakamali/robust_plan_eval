import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import scipy.stats as st
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from IPython.display import display

def describe_plot(arrays,titles,metric_label,plot_title,log_scale = True,figsize=[10,6],plot_x_every=5,save_to=None,bbox_to_anchor=None,show_fig=True):
    arrays = np.array(arrays).T
    data = pd.DataFrame(arrays,columns=titles)
    data = data.describe(percentiles=np.arange(0,1,0.01))
    plot_dist_line(data,metric_label,plot_title,log_scale,figsize=figsize,
    plot_x_every=plot_x_every,save_to=save_to,bbox_to_anchor=bbox_to_anchor,show_fig=show_fig)

def plot_step_curve(arrays,titles,scale='log',title='Workload Runtime',mask=None,figsize=[10,6],
                    save_to=None,bbox_to_anchor=None,show_fig=True):
    fig = plt.figure(figsize=figsize)
    for idx,array in enumerate(arrays):
        if mask is not None:
            array = array[mask]
        # array = np.sort(array)
        x = np.arange(array.shape[0])
        y = array.cumsum()
        plt.step(x, y + 2, label=titles[idx])
    plt.yscale(scale)
    plt.grid(axis='x', color='0.95')
    plt.legend(bbox_to_anchor=bbox_to_anchor)
    plt.title(title)
    plt.xlabel('Queries')
    plt.ylabel('Runtime (s)')
    if save_to is not None:
        plt.savefig(save_to,bbox_inches='tight',dpi=300)
    if show_fig:
        plt.show()

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


def compare2methods(runtime1,runtime2):
    assert (runtime1.shape[0] == runtime2.shape[0])
    imp = (runtime1/runtime2)<1
    reg = (runtime1/runtime2)>1
    unch = (runtime1/runtime2)==1
    results = {}

    results['improved'] = imp.mean()
    if results['improved'] == 0:
        results['improved_runtime'] = 0
        results['average_improvement'] = 0
    else:
        results['improved_runtime'] = (runtime2-runtime1)[imp].sum()#/runtime2[imp].sum()
        results['average_improvement'] = (runtime2-runtime1)[imp].mean()#sum()/imp.sum()

    results['regressed'] = reg.mean()
    if results['regressed'] == 0:
        results['regressed_runtime'] = 0
        results['average_regression'] = 0
    else:
        results['regressed_runtime'] = (runtime1-runtime2)[reg].sum()#/runtime2[reg].sum()
        results['average_regression'] = (runtime1-runtime2)[reg].mean()#sum()/reg.sum()

    results['unchanged'] = unch.mean()
    
    results['total_runtime_change%'] = (runtime1-runtime2).sum()/runtime2.sum()
    results['total_runtime_change'] = (runtime1-runtime2).sum()
    results['average_runtime_change'] = (runtime1-runtime2).mean()
    
    results['total_subopt'] = runtime1.sum()/runtime2.sum()
    
    return results

def comput_uncertainty(raw_preds):
    m = raw_preds.mean(axis=0).numpy()[:,0]
    Ud = raw_preds.mean(axis=0).numpy()[:,1]
    Um = raw_preds.var(axis=0).numpy()[:,0]
    return m,Ud,Um

def find_optimal_cons_ratio(m,s2,val_set,ratios = np.arange(0.0,1,0.1),minimize='runtime'):
    val_cons_rt_ratio = []
    val_cons_so_ratio = []
    for ratio in ratios:
        strategy_args = {'cons_ratio':ratio}
        val_cons_subopts, val_cons_runtimes,_ = evaluate_method(m,s2,val_set, strategy='conservative',**strategy_args)
        val_cons_rt_ratio.append(val_cons_runtimes)
        val_cons_so_ratio.append(val_cons_subopts)
    subopt_per_ratio = np.array(val_cons_so_ratio).mean(axis=1)
    rt_per_ratio = np.array(val_cons_rt_ratio).mean(axis=1)
    if minimize =='runtime':
        optimal_ratio = ratios[np.argmin(rt_per_ratio)]
    else:
        optimal_ratio = ratios[np.argmin(subopt_per_ratio)]
    # plot_step_curve(val_cons_rt_ratio,['cons_ratio: '+str(i) for i in ratios],show_fig=show_fig)
    print('optimal_ratio',optimal_ratio)
    return optimal_ratio

def find_optimal_prune_ratio(p1,p2,m,s2,val_set,strategy='baseline ml',
                             minimize ='runtime',
                             p1_grid = np.array([0]),
                             p2_grid = np.array([0]),
                            strategy_args={}):
    strategy_args=strategy_args.copy()
    rt_ratio = []
    so_ratio = []
    rt_ratio = np.zeros((p1_grid.shape[0],p2_grid.shape[0]))
    so_ratio = np.zeros((p1_grid.shape[0],p2_grid.shape[0]))
    for i, r_d in enumerate(p1_grid):
        for j, r_m in enumerate(p2_grid):
            strategy_args[p1]=r_d 
            strategy_args[p2]=r_m
            subopts,runtimes,_ = evaluate_method(m,s2,
                         val_set, strategy=strategy, prune=True,
                         **strategy_args)
            rt_ratio[i,j] = runtimes.mean()
            so_ratio[i,j] = subopts.mean()

    if minimize =='runtime':
        opt_target = rt_ratio
    else:
        opt_target = so_ratio
    
    print(opt_target)
        
    ind = np.unravel_index(np.argmin(opt_target), opt_target.shape)
    optimal_ratio = p1_grid[ind[0]],p2_grid[ind[1]]
    
    strategy_args[p1]=optimal_ratio[0]
    strategy_args[p2]=optimal_ratio[1]
    print('optimal strategy arguments',strategy_args)
    
    return strategy_args

# Compare pairs of methods

def compare_with_baseline(values, labels):
    results = {}
    for idx1, val1 in enumerate(values):
        for idx2, val2 in enumerate(values):
            if idx1 == 0 and idx2 != idx1:
                res = compare2methods(val2,val1)
                results[labels[idx2]] = res
    return pd.DataFrame(results).T

def comparetopplansmulti(bm,bs2,p,verbose=False):
    # p: number of top plans to copare
    # bm: expected exec time
    # bs2: expected variance
    N = bm.shape[0]
    minS2 = bs2[bs2>0].min()
    bs2[bs2<=0]=minS2
    if verbose:
        print("best_m",bm)
        print("best_s2",bs2)
        print("best_s",np.sqrt(bs2))
    M = np.zeros((N,p,p)) # m(i) - m(j)
    S = np.zeros((N,p,p)) # sqrt(s2(i) + s2(j))
    Z = np.zeros((N,p,p)) # z-score = (0-M)/S
    for n in range(N):
        # print("query",n)
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

def evaluate_method(pred, uncertainty, data_obj, strategy, 
                    prune=False, volatility=None, **kwargs):

    tic = time.time()

    strategies = ['baseline ml','baseline opt','conservative','subopt risk']
    # sanity check for the value passed to the 'strategy' argument
    if strategy not in strategies:
        exception_msg = 'strategy must be in: \'{}\''.format('\',\''.join(strategies))
        raise ValueError(exception_msg)

    # supports either one or two uncertainty values
    if isinstance(uncertainty, list):
        data_unc = uncertainty[0]
        model_unc = uncertainty[1]
        total_unc = data_unc + model_unc
    else:
        data_unc = uncertainty
        model_unc = uncertainty
        total_unc = uncertainty
    
    if volatility is not None:
            volatility = volatility.reshape(-1,1)
    
    # capture the optimizer cost to be used for strategy = 'baseline opt' or if the search space is null after pruning
    opt_cost = data_obj.opt_cost
    y = data_obj.y
    y_t = data_obj.y_t


    strategy_runtimes=[]
    best_runtimes=[]
    strategy_plan_volatilities=[]
    corr_all=[]
    target_super=[]
    runtime_super=[]
    count_fail = 0
    query_ids = np.array(data_obj.query_id)
    for _, q in enumerate(np.unique(query_ids)):
        msk = (query_ids == q)
        num_plans = msk.sum()
        pred_sub = pred[msk].reshape(-1,1)
        data_unc_sub = data_unc[msk].reshape(-1,1)
        model_unc_sub = model_unc[msk].reshape(-1,1)
        total_unc_sub = total_unc[msk].reshape(-1,1)
        opt_cost_sub = np.log10(opt_cost[msk].reshape(-1,1).numpy())
        runtime = y[msk].reshape(-1,1).numpy()
        runtime_t = y_t[msk].reshape(-1,1).numpy()

        if volatility is not None:
            plan_volatility = volatility[msk].reshape(-1,1)
        
        prune_msk = np.array([True]*msk.sum()).reshape(-1,1)
        if prune:
            # if prune_ratio is given it will be used for both model and data prune ratios
            if 'prune_ratio' in kwargs:
                kwargs['data_prune_ratio'] = kwargs['prune_ratio']
                kwargs['model_prune_ratio'] = kwargs['prune_ratio']
            
            if 'data_prune_ratio' in kwargs:
                plans_to_keep = int(num_plans*(1-kwargs['data_prune_ratio']))-1
                
                if plans_to_keep == -1:
                    thr = 0
                else:
                    thr = data_unc_sub[np.argsort(data_unc_sub,axis=0)[plans_to_keep]]

                msk1 = (data_unc_sub <= thr)
                prune_msk = prune_msk & msk1
                
            if 'model_prune_ratio' in kwargs:
                plans_to_keep = int(num_plans*(1-kwargs['model_prune_ratio']))-1
                thr = model_unc_sub[np.argsort(model_unc_sub,axis=0)[plans_to_keep]]
                msk2 = (model_unc_sub <= thr)
                prune_msk = prune_msk & msk2
            
            if 'max_data_uncertainty' in kwargs:
                msk3 = (data_unc_sub <= kwargs['max_data_uncertainty'])
                prune_msk = prune_msk & msk3

            if 'max_model_uncertainty' in kwargs:
                msk4 = (model_unc_sub <= kwargs['max_model_uncertainty'])
                prune_msk = prune_msk & msk4

        runtime_sub=runtime[prune_msk]
        runtime_t_sub=runtime_t[prune_msk]

        # if search space is null after pruning, fall back on optimizer's plan 
        if prune_msk.sum() == 0:
            target_sub=opt_cost_sub.squeeze()
            runtime_sub=runtime.squeeze()
            runtime_t_sub=runtime_t.squeeze()
            # print('fall back on optimizer')
            if volatility is not None:
                plan_volatility_sub = plan_volatility
            count_fail+=1
        
        else:
            if strategy == strategies[1]:
                target_sub = opt_cost_sub[prune_msk]
            elif strategy == strategies[0]:
                target_sub = pred_sub[prune_msk]
            elif strategy == strategies[2]:
                target_sub = pred_sub[prune_msk]+np.sqrt(total_unc_sub[prune_msk])*kwargs['cons_ratio']
            elif strategy == strategies[3]:
                num_plans_fin = prune_msk.sum()
                m = pred_sub[prune_msk].reshape(-1,num_plans_fin)
                s2 = total_unc_sub[prune_msk].reshape(-1,num_plans_fin)
                target_sub = comparetopplansmulti(m,s2,num_plans_fin).reshape(-1)
            if volatility is not None:
                plan_volatility_sub = plan_volatility[prune_msk]
        
        # if the sub has only one element with no dimensions, add dimension so that the following operations do not fail
        target_sub=target_sub.reshape(-1)
        runtime_sub=runtime_sub.reshape(-1)
        runtime_t_sub=runtime_t_sub.reshape(-1)

        target_super.extend(target_sub.tolist())
        runtime_super.extend(runtime_t_sub.tolist())

        strategy_plan = np.argmin(target_sub)
        strategy_runtime = runtime_sub[strategy_plan]

        if volatility is not None:
            strategy_plan_volatility = plan_volatility_sub[strategy_plan].item()
            strategy_plan_volatilities.append(strategy_plan_volatility)

        best_plan = np.argmin(runtime)
        best_runtime = runtime[best_plan]
        strategy_runtimes.append(strategy_runtime.item())
        best_runtimes.append(best_runtime.item())
    target_super = np.array(target_super).squeeze()
    runtime_super = np.array(runtime_super).squeeze()
    corr_all = np.corrcoef(target_super,runtime_super)[0,1]
    strategy_runtimes=np.array(strategy_runtimes).squeeze()
    best_runtimes=np.array(best_runtimes).squeeze()
    subopts = strategy_runtimes/best_runtimes
    
    toc = time.time()

    evaluation_time = toc - tic

    prune_label = 'prune' if prune else ''
    print(strategy,prune_label,': corr :',corr_all)
    print('queries with no plans after pruning:',count_fail)
    print('evaluation time:', evaluation_time)
    print('')
    
    if volatility is not None:
        strategy_plan_volatilities=np.array(strategy_plan_volatilities).reshape(-1)
        return subopts,strategy_runtimes,best_runtimes,strategy_plan_volatilities
    else:
        return subopts,strategy_runtimes,best_runtimes


def lcm_model_eval(test_set= None, ypreds_tens_test_org= None, 
               dataset_label = None, model_label = None,
              percentiles = np.arange(0,1,0.1),load_from_disk=False, files_id = None,show_fig=True):
    import torch
    if load_from_disk:
        q_error_tensor,q_error_alt_tensor,num_joins_qe_tensor,ml_subOptTensor, db2_subOptTensor, num_joins_so_tensor = torch.load('./results/{}_{}_{}.pt'.format(model_label,dataset_label,files_id))
    else:
        from util.torch_scorers import subOpt2, q_error, q_error_alt
        ml_subOptList=[]
        db2_subOptList=[]
        q_error_list = []
        q_error_alt_list = []
        num_joins_list = []
        num_joins_qe_list = []
        query_ids = np.array(test_set.query_id)
        for query_id in np.unique(query_ids):
            msk = (query_ids == query_id)
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

def compute_spearmanr(x,y,verbose=True):
    spearmanr_coef, p_value = spearmanr(x,y)
    # Display the results
    if verbose:
        print(f"Spearman's Rank Correlation Coefficient: {spearmanr_coef}")
        print(f"P-value: {p_value}")

        # Interpret the results
        if p_value < 0.05:
            print("There is a significant correlation.")
        else:
            print("There is no significant correlation.")
    return spearmanr_coef, p_value

def compute_pearsonr(x,y,verbose=True):
    # Calculate Pearson correlation coefficient and p-value
    temp = pearsonr(x,y)
    correlation_coefficient, p_value = temp
    # Display the results
    if verbose:
        print(f"Pearson Correlation Coefficient: {correlation_coefficient}")
        print(f"Confidence interval with 95% confidence level: {temp.confidence_interval(.95)}")
        print(f"P-value: {p_value}")

        # Interpret the results
        if p_value < 0.05:
            print("There is a significant correlation.")
        else:
            print("There is no significant correlation.")
    return correlation_coefficient, p_value

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