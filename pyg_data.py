# TODO: Make sampling test from long running optional 

import os
import pickle
import math
from itertools import compress, combinations
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset
from util.tcnn_util import prepare_trees, transformer, left_child, right_child
from IPython.display import clear_output
import random

log_run_tail_thr = 97

class queryPlanPGDataset(InMemoryDataset):
    def __init__(self, root='./',  split: str = "train", 
                 transform=None, pre_transform=None, 
                 pre_filter=None, force_reload = False,files_id=None, labeled_data_dir='./labeled_data/',
                 seed = 0, num_samples = None, val_samples = 0.1, test_samples = 0.01, test_slow_samples=0.5):
        self.files_id = files_id
        self.labeled_data_dir = labeled_data_dir
        self.seed = seed
        self.num_samples = num_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.test_slow_samples = test_slow_samples

        super().__init__(root, transform, pre_transform, pre_filter,force_reload=force_reload)

        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        else:
            raise ValueError(f"Split '{split}' found, but expected either "
                             f"'train', 'val', or 'test'")

        self.data, self.slices = torch.load(path, weights_only=False)
            
    @property
    def raw_file_names(self):
        return [os.path.join(self.labeled_data_dir,'labeled_query_plans_{}.pickle'.format(self.files_id))
               ]

    @property
    def processed_file_names(self):
        return [
            'proc_data_{}_tr.pt'.format(self.files_id),
            'proc_data_{}_val.pt'.format(self.files_id),
            'proc_data_{}_ts.pt'.format(self.files_id),
               ]

    def process(self):
        # Read data into huge `Data` list.
        with open(self.raw_file_names[0],'rb') as f:
            queries_list = pickle.load(f)
        
        indices = [i for i in range(len(queries_list))]
        
        if self.num_samples is not None:
            np.random.seed(self.seed)
            sample = np.random.choice(indices, size = self.num_samples, replace = False)
            queries_list=queries_list[sample]
        fin_samples = len(queries_list)

        # this loop prepares the plan trees
        plan_trees = []
        for query in queries_list:
            for j in query.plans.keys():
                plan = query.plans[j]
                plan_trees.append(plan.plan_tree)

        # prepare plan trees for TCNN
        prep_trees=prepare_trees(plan_trees, transformer, left_child, right_child, cuda=False)
        
        # create individual Data objects for each single sample
        # then put them in data_list
        data_list = []
        query_ids = []
        opt_plan_msk = []
        plan_latency_list = []
        prep_tree_id = 0
        for i,query in enumerate(queries_list):
            if i%5 == 0:
                clear_output(wait=True)
                print('Loading the data ... {}%'.format(math.floor((i/fin_samples)*100)))
            for j in query.plans.keys():
                plan = query.plans[j]
                
                # skip the plan if the plan's cost or latency are not populated
                if plan.latency is None or plan.cost.size == 0:
                    break

                # if int(query.edge_indc.shape[1]/2) > 6:
                #     break
                
                # replace the latency for timed out plans with 5*opt_plan_latency 
                opt_plan = False
                plan_latency = plan.latency
                if plan.hintset_id == 0:
                    opt_plan = True
                    opt_plan_latency = plan.latency
                else:
                    if plan.timed_out == True:
                        plan_latency = 10*opt_plan_latency
                    
                opt_plan_msk.append(opt_plan)
                plan_latency_list.append(plan_latency)

                # assign plan tree attributes and orders
                prep_tree_attr = prep_trees[0][prep_tree_id]
                prep_tree_ord = prep_trees[1][prep_tree_id]
                prep_tree_id+=1

                data = Data(
                    x_s=torch.Tensor(query.node_attr),
                    edge_index_s=torch.Tensor(query.edge_indc),
                    edge_attr_s =torch.Tensor(query.edge_attr),
                    graph_attr = torch.Tensor(query.graph_attr),
                    y = torch.Tensor([float(plan_latency)]),
                    plan_attr=torch.Tensor(prep_tree_attr),
                    plan_ord=torch.Tensor(prep_tree_ord),
                    query_id = query.q_id,
                    num_joins = torch.Tensor([int(query.edge_indc.shape[1]/2)]),
                    opt_choice = torch.Tensor([opt_plan]),
                    opt_cost = torch.Tensor([float(plan.cost)]),
                    y_t = torch.Tensor([float(plan_latency)]),  # placeholder for transformed targets
                    num_nodes = torch.Tensor(query.node_attr).shape[0], 
                    # purturbed_runtimes = torch.Tensor(purturbed_runtimes[i]),
                    )
                query_ids.append(query.q_id)
                data_list.append(data)
        
        clear_output(wait=True)
        print('Loading the data ... {}%'.format(str(100)))
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        # convert lists to arrays
        plan_latency_list = np.array(plan_latency_list)
        opt_plan_msk = np.array(opt_plan_msk)
        query_ids = np.array(query_ids)

        # get unique query ids
        query_ids_unique = pd.unique(query_ids)

        # transform ratio sample sizes to numbers
        if self.test_samples < 1:
            self.test_samples = int(self.test_samples*query_ids_unique.shape[0])
        if self.val_samples < 1:
            self.val_samples = int(self.val_samples*query_ids_unique.shape[0])

        # get query ids for long running queries
        opt_plan_latency=plan_latency_list[opt_plan_msk]
        long_running_msk = (opt_plan_latency>np.percentile(opt_plan_latency,q=log_run_tail_thr))
        long_running_qids = query_ids_unique[long_running_msk]
        non_long_qids = query_ids_unique[~long_running_msk]

        # determine the share of test data from long-running queries
        if self.test_slow_samples is None:
            self.test_slow_samples = self.test_samples/(query_ids_unique.size)
        
        # sample part 1 from long running queries
        train_val_qid1, test_qid1 = train_test_split(
            long_running_qids,test_size=self.test_slow_samples,
            random_state=self.seed, shuffle=True
            ) # by default gets 1/2 of all long running queries

        test_samples2 = max(int(self.test_samples - test_qid1.shape[0]),1)

        # sample part 2 from non-long running queries
        train_val_qid2, test_qid2 = train_test_split(
            non_long_qids,
            test_size=test_samples2,
            random_state=self.seed, shuffle=True
            )
        
        # get the unions for short/long running query ids
        test_qid = np.union1d(test_qid1,test_qid2)
        train_val_qid = np.union1d(train_val_qid1,train_val_qid2)

        # get val split query ids
        train_qid, val_qid=train_test_split(
            train_val_qid,
            test_size=self.val_samples,
            random_state=self.seed, shuffle=True
            )
        
        # get split slices
        train_slice = np.isin(query_ids, train_qid)
        val_slice = np.isin(query_ids, val_qid)
        test_slice = np.isin(query_ids, test_qid)
        
        tr_data, tr_slices = self.collate(list(compress(data_list, train_slice)))
        torch.save((tr_data, tr_slices), self.processed_paths[0])
        
        val_data, val_slices = self.collate(list(compress(data_list, val_slice)))
        torch.save((val_data, val_slices), self.processed_paths[1])
        
        ts_data, ts_slices = self.collate(list(compress(data_list, test_slice)))

        torch.save((ts_data, ts_slices), self.processed_paths[2])


class queryPlanPGDataset_nosplit(InMemoryDataset):
    def __init__(self, root='./', 
                 transform=None, pre_transform=None, 
                 pre_filter=None, force_reload = False,files_id=None, labeled_data_dir='./labeled_data/',
                 seed = 0, num_samples = None):
        self.files_id = files_id
        self.labeled_data_dir = labeled_data_dir
        self.seed = seed
        self.num_samples = num_samples

        super().__init__(root, transform, pre_transform, pre_filter,force_reload=force_reload)

        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path, weights_only=False)
            
    @property
    def raw_file_names(self):
        return [os.path.join(self.labeled_data_dir,'labeled_query_plans_{}.pickle'.format(self.files_id))
               ]

    @property
    def processed_file_names(self):
        return [
            'proc_data_{}.pt'.format(self.files_id),
               ]

    def process(self):
        # Read data into huge `Data` list.
        with open(self.raw_file_names[0],'rb') as f:
            queries_list = pickle.load(f)
        
        if self.num_samples is not None:
            np.random.seed(self.seed)
            queries_list = np.random.choice(queries_list, size = self.num_samples, replace = False)
            print("sampling {} queries",len(queries_list))
        fin_samples = len(queries_list)

        # this loop prepares the plan trees
        plan_trees = []
        for query in queries_list:
            for j in query.plans.keys():
                plan = query.plans[j]
                plan_trees.append(plan.plan_tree)

        # prepare plan trees for TCNN
        prep_trees=prepare_trees(plan_trees, transformer, left_child, right_child, cuda=False)
        
        # create individual Data objects for each single sample
        # then put them in data_list
        data_list = []
        query_ids = []
        opt_plan_msk = []
        plan_latency_list = []
        prep_tree_id = 0
        for i,query in enumerate(queries_list):
            if i%5 == 0:
                clear_output(wait=True)
                print('Loading the data ... {}%'.format(math.floor((i/fin_samples)*100)))
            for j in query.plans.keys():
                plan = query.plans[j]
                
                # skip the plan if the plan's cost or latency are not populated
                if plan.latency is None or plan.cost.size == 0:
                    break
                
                # replace the latency for timed out plans with 10*opt_plan_latency 
                opt_plan = False
                plan_latency = plan.latency
                if plan.hintset_id == 0:
                    opt_plan = True
                    opt_plan_latency = plan.latency
                else:
                    if plan.timed_out == True:
                        plan_latency = 10*opt_plan_latency
                    
                opt_plan_msk.append(opt_plan)
                plan_latency_list.append(plan_latency)

                # assign plan tree attributes and orders
                prep_tree_attr = prep_trees[0][prep_tree_id]
                prep_tree_ord = prep_trees[1][prep_tree_id]
                prep_tree_id+=1

                data = Data(
                    x_s=torch.Tensor(query.node_attr),
                    edge_index_s=torch.Tensor(query.edge_indc),
                    edge_attr_s =torch.Tensor(query.edge_attr),
                    graph_attr = torch.Tensor(query.graph_attr),
                    y = torch.Tensor([float(plan_latency)]),
                    plan_attr=torch.Tensor(prep_tree_attr),
                    plan_ord=torch.Tensor(prep_tree_ord),
                    query_id = query.q_id,
                    num_joins = torch.Tensor([int(query.edge_indc.shape[1]/2)]),
                    opt_choice = torch.Tensor([opt_plan]),
                    opt_cost = torch.Tensor([float(plan.cost)]),
                    y_t = torch.Tensor([float(plan_latency)]),  # placeholder for transformed targets
                    num_nodes = torch.Tensor(query.node_attr).shape[0], 
                    # purturbed_runtimes = torch.Tensor(purturbed_runtimes[i]),
                    )
                query_ids.append(query.q_id)
                data_list.append(data)
        
        clear_output(wait=True)
        print('Loading the data ... {}%'.format(str(100)))
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        # convert lists to arrays
        plan_latency_list = np.array(plan_latency_list)
        opt_plan_msk = np.array(opt_plan_msk)
        query_ids = np.array(query_ids)

        # get unique query ids
        query_ids_unique = pd.unique(query_ids)
        
        # get split slices
        train_slice = np.isin(query_ids, query_ids_unique)
        
        tr_data, tr_slices = self.collate(list(compress(data_list, train_slice)))
        torch.save((tr_data, tr_slices), self.processed_paths[0])

def get_multipred_slice(query_ids):
    mp_ids = [id for id in query_ids if 'mp' in id]
    mp_ids = np.array(mp_ids)
    mp_ids = np.unique(mp_ids)
    return np.isin(query_ids, mp_ids)

def get_benchmark_slice(query_ids):
    benchmark_ids = [id for id in query_ids if 'q' not in id and 'mp' not in id]
    benchmark_ids = np.array(benchmark_ids)
    benchmark_ids = np.unique(benchmark_ids)
    return np.isin(query_ids, benchmark_ids)

class queryPlanPGDataset_withbenchmark(InMemoryDataset):
    def __init__(self, root='./',  split: str = "train", 
                 transform=None, pre_transform=None, 
                 pre_filter=None, force_reload = False,files_id=None,proc_files_id=None, labeled_data_dir='./labeled_data/',
                 seed = 0, num_samples = None, val_samples = 0.1, test_samples = 0.01, test_slow_samples=0.5, benchmark_files_id = '',
                 exp_id=''):
        self.files_id = files_id
        self.proc_files_id = proc_files_id if proc_files_id is not None else files_id
        self.labeled_data_dir = labeled_data_dir
        self.seed = seed
        self.num_samples = num_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.test_slow_samples = test_slow_samples
        self.benchmark_files_id = benchmark_files_id
        self.exp_id = exp_id

        super().__init__(root, transform, pre_transform, pre_filter,force_reload=force_reload)

        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        elif split == 'benchmark':
            path = self.processed_paths[3]
        else:
            raise ValueError(f"Split '{split}' found, but expected either "
                             f"'train', 'val', 'test' or 'benchmark'")

        self.data, self.slices = torch.load(path, weights_only=False)
            
    @property
    def raw_file_names(self):
        return [
            os.path.join(self.labeled_data_dir,'labeled_query_plans_{}.pickle'.format(self.files_id)),
            os.path.join(self.labeled_data_dir,'labeled_query_plans_{}.pickle'.format(self.benchmark_files_id))
            ]

    @property
    def processed_file_names(self):
        return [
            'proc_data_{}_{}_tr.pt'.format(self.proc_files_id,self.exp_id),
            'proc_data_{}_{}_val.pt'.format(self.proc_files_id,self.exp_id),
            'proc_data_{}_{}_ts.pt'.format(self.proc_files_id,self.exp_id),
            'proc_data_{}_{}_bm.pt'.format(self.proc_files_id,self.exp_id),
            ]

    def process(self):
        # Read data into huge `Data` list.
        with open(self.raw_file_names[0],'rb') as f:
            queries_list = pickle.load(f)

        with open(self.raw_file_names[1],'rb') as f:
            bm_queries_list = pickle.load(f)
        
        # sample the raw dataset if self.num_samples is provided
        if self.num_samples is not None:
            random.seed(self.seed)
            queries_list = random.sample(queries_list, k = self.num_samples)
        
        # get query ids
        query_ids_unique = np.array([q.q_id for q in queries_list])
        benchmark_ids = np.array([q.q_id for q in bm_queries_list])
        queries_superset = queries_list + bm_queries_list
        fin_samples = len(queries_superset)

        # prepare plan trees for TCNN
        plan_trees = []
        plan_trees_ref = {}
        for i,query in enumerate(queries_superset):
            for j in query.plans.keys():
                plan = query.plans[j]
                plan_trees.append(plan.plan_tree)
                plan_trees_ref[str(i)+'-'+str(j)]=len(plan_trees)-1
        prep_trees = prepare_trees(plan_trees, transformer, left_child, right_child, cuda=False)
        
        # create individual Data objects for each single sample
        # then put them in data_list
        data_list = []
        query_ids = []
        opt_plan_latencies = []
        for i,query in enumerate(queries_superset):
            if i%5 == 0:
                clear_output(wait=True)
                print('Loading the data ... {}%'.format(math.floor((i/fin_samples)*100)))
            
            opt_plan = query.plans[0]

            if opt_plan.latency is None or opt_plan.cost.size == 0:
                print(">>>>>>>>>>>>>skip query {} completely as the default plan is not labeled properly!".format(i))
                continue

            # extract query_id, optimizer plan latency pair for non-benchmark queries
            opt_plan_latency = opt_plan.latency
            opt_plan_latencies.append([query.q_id,opt_plan.latency])

            for j in query.plans.keys():
                plan = query.plans[j]
                
                # skip the plan if the plan's cost or latency are not populated
                if plan.latency is None or plan.cost.size == 0:
                    print(">>>>>>>>>>>>>skip query {}, plan {}".format(i,j))
                    continue

                plan_latency = plan.latency

                # replace the latency for timed out plans with 10*opt_plan_latency 
                opt_choice = False
                if plan.hintset_id == 0:
                    opt_choice = True
                else:
                    if plan.timed_out == True:
                        plan_latency = 10*opt_plan_latency
                    

                # assign plan tree attributes and orders
                plan_ref_idx = plan_trees_ref[str(i)+'-'+str(j)]
                prep_tree_attr = prep_trees[0][plan_ref_idx]
                prep_tree_ord = prep_trees[1][plan_ref_idx]

                data = Data(
                    x_s=torch.Tensor(query.node_attr),
                    edge_index_s=torch.Tensor(query.edge_indc),
                    edge_attr_s =torch.Tensor(query.edge_attr),
                    graph_attr = torch.Tensor(query.graph_attr),
                    y = torch.Tensor([float(plan_latency)]),
                    plan_attr=torch.Tensor(prep_tree_attr),
                    plan_ord=torch.Tensor(prep_tree_ord),
                    query_id = query.q_id,
                    num_joins = torch.Tensor([int(query.edge_indc.shape[1]/2)]),
                    opt_choice = torch.Tensor([opt_choice]),
                    opt_cost = torch.Tensor([float(plan.cost)]),
                    y_t = torch.Tensor([float(plan_latency)]),  # placeholder for transformed targets
                    num_nodes = torch.Tensor(query.node_attr).shape[0], 
                    )
                query_ids.append(query.q_id)
                data_list.append(data)
        
        clear_output(wait=True)
        print('Loading the data ... {}%'.format(str(100)))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        # convert lists to arrays
        query_ids = np.array(query_ids)
        opt_plan_latencies = np.array(opt_plan_latencies) # unique per query

        # get a slice that contains the benchmark query ids 
        benchmark_slice = np.isin(query_ids, benchmark_ids)

        # Implement split logic
        query_ids_unique = np.array(query_ids_unique)  # non-benchmark query ids
        np.random.seed(self.seed)  # Set the random seed

        # transform ratio sample sizes to numbers
        if self.test_samples < 1:
            self.test_samples = int(self.test_samples*query_ids_unique.shape[0])
        if self.val_samples < 1:
            self.val_samples = int(self.val_samples*query_ids_unique.shape[0])

        # Separate query IDs and latencies into two lists
        opt_plan_q_ids, opt_plan_latencies = zip(*[
            (q_id, latency) for q_id, latency in opt_plan_latencies if q_id in query_ids_unique
        ])

        # Convert latencies to a numpy array for percentile calculation
        opt_plan_latencies = np.array(opt_plan_latencies, dtype=float)

        # Calculate the long-running threshold
        long_run_threshold = np.percentile(opt_plan_latencies, log_run_tail_thr)

        # Identify long-running queries
        long_running_queries = np.array(opt_plan_q_ids)[opt_plan_latencies > long_run_threshold]
        non_long_running_queries = np.array(opt_plan_q_ids)[opt_plan_latencies <= long_run_threshold]
        print("long_running_queries",long_running_queries.shape[0])

        # Assign long-running queries to the test set (randomly sample)
        num_long_run_test_samples = int(len(long_running_queries) * self.test_slow_samples)
        test_long_run_qids = np.random.choice(long_running_queries, size=num_long_run_test_samples, replace=False)
        print("test_long_run_qids",test_long_run_qids.shape[0])


        # Remaining test set size
        remaining_test_samples = self.test_samples - len(test_long_run_qids)
        print("remaining_test_samples", remaining_test_samples)
        print("query_ids_unique",query_ids_unique)
        remaining_test_samples = max(0, remaining_test_samples)  # Ensure non-negative

        # Sample remaining test set from non-long-running queries
        test_non_long_run_qids = np.random.choice(non_long_running_queries, size=remaining_test_samples, replace=False)

        # Combine all test set queries
        test_qid = np.concatenate((test_long_run_qids, test_non_long_run_qids))

        # Remaining queries (excluding test set) for training and validation
        remaining_qids = np.setdiff1d(query_ids_unique, test_qid)

        # Split remaining queries into training and validation sets
        val_qid = np.random.choice(remaining_qids, size=self.val_samples, replace=False)
        train_qid = np.setdiff1d(remaining_qids, val_qid)

        # get split slices
        train_slice = np.isin(query_ids, train_qid)
        val_slice = np.isin(query_ids, val_qid)
        test_slice = np.isin(query_ids, test_qid)
        
        tr_data, tr_slices = self.collate(list(compress(data_list, train_slice)))
        torch.save((tr_data, tr_slices), self.processed_paths[0])
        
        val_data, val_slices = self.collate(list(compress(data_list, val_slice)))
        torch.save((val_data, val_slices), self.processed_paths[1])
        
        ts_data, ts_slices = self.collate(list(compress(data_list, test_slice)))
        torch.save((ts_data, ts_slices), self.processed_paths[2])

        bm_data, bm_slices = self.collate(list(compress(data_list, benchmark_slice)))
        torch.save((bm_data, bm_slices), self.processed_paths[3])
