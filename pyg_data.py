import os
import pickle
import math
from itertools import compress
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset
from util.tcnn_util import prepare_trees, transformer, left_child, right_child

class queryPlanPGDataset(InMemoryDataset):
    def __init__(self, root='./',  split: str = "train", 
                 transform=None, pre_transform=None, 
                 pre_filter=None, force_reload = False,files_id=None, labeled_data_dir='./internal/',
                 seed = 0, num_samples = None, val_samples = 0.1, test_samples = 0.1):
        self.files_id = files_id
        self.labeled_data_dir = labeled_data_dir
        self.seed = seed
        self.num_samples = num_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

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

        self.data, self.slices = torch.load(path)
            
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

#     def download(self):
#         # Download to `self.raw_dir`.
#         download_url(url, self.raw_dir)
#         ...

    def process(self):
        # Read data into huge `Data` list.
        with open(self.raw_file_names[0],'rb') as f:
            queries_list = pickle.load(f)
        
        indices = [i for i in range(len(queries_list))]
        
        if self.num_samples is not None:
            sample = np.random.choice(indices, size = self.num_samples, replace = False)
            queries_list=queries_list[sample]
        fin_samples = len(queries_list)

        # this loop prepares the plan trees
        plan_trees = []
        for i in range(fin_samples):
            query = queries_list[i]
            for j in query.plans.keys():
                plan = query.plans[j]
                plan_trees.append(plan.plan_tree)
        # print("ql ------------->",queries_list[0].plans[0].guideline)
        # print("ql ------------->",queries_list[0].plans[0].plan_tree.print())
        prep_trees=prepare_trees(plan_trees, transformer, left_child, right_child, cuda=False)
        
        # create individual Data objects for each single sample
        # then put them in data_list
        data_list = []
        query_ids = []
        prep_tree_id = 0
        for i in range(fin_samples):
            query = queries_list[i]
            query_ids.append(query.q_id)
            if i%5 == 0:
                print('Loading the data ... {}%'.format(math.floor((i/fin_samples)*100)))
            for j in query.plans.keys():
                
                plan = query.plans[j]
                plan_cost = plan.cost if plan.cost !='' else 0
                
                prep_tree_attr = prep_trees[0][prep_tree_id]
                prep_tree_ord = prep_trees[1][prep_tree_id]
                prep_tree_id+=1
                
                opt_plan = False
                if plan.hintset_id == 0:
                    opt_plan = True
                
                data = Data(
                    x_s=torch.Tensor(query.node_attr),
                    edge_index_s=torch.Tensor(query.edge_indc),
                    edge_attr_s =torch.Tensor(query.edge_attr),
                    graph_attr = torch.Tensor(query.graph_attr),
                    y = torch.Tensor([plan.latency]),
                    plan_attr=torch.Tensor(prep_tree_attr),
                    plan_ord=torch.Tensor(prep_tree_ord),
                    query_id = query.q_id,
                    num_joins = torch.Tensor([int(query.edge_attr.shape[1]/2)]),
                    opt_choice = torch.Tensor([opt_plan]),
                    opt_cost = torch.Tensor([float(plan_cost)]),
                    y_t = torch.Tensor([plan.latency]),  # placeholder for transformed targets
                    num_nodes = torch.Tensor(query.node_attr).shape[0], 
                    # purturbed_runtimes = torch.Tensor(purturbed_runtimes[i]),
                    )
                data_list.append(data)

        print('Loading the data ... {}%'.format(100))
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        # get test and validation splits from samples that have purturbed runtimes captured for evaluation purposes
        query_ids = np.array(query_ids)
        train_val_qid, test_qid = train_test_split(query_ids,
                        test_size=self.test_samples,
                        random_state=self.seed, shuffle=True)
        train_qid, val_qid=train_test_split(train_val_qid, 
                        test_size=self.val_samples,
                        random_state=self.seed, shuffle=True)
        
        train_slice = (np.isin(query_ids, train_qid))
        val_slice = (np.isin(query_ids, val_qid))
        test_slice = (np.isin(query_ids, test_qid))
        
        tr_data, tr_slices = self.collate(list(compress(data_list, train_slice)))
        torch.save((tr_data, tr_slices), self.processed_paths[0])
        
        val_data, val_slices = self.collate(list(compress(data_list, val_slice)))
        torch.save((val_data, val_slices), self.processed_paths[1])
        
        ts_data, ts_slices = self.collate(list(compress(data_list, test_slice)))
        torch.save((ts_data, ts_slices), self.processed_paths[2])

# train_set = queryPlanPGDataset(root = './', split= 'train')
# val_set = queryPlanPGDataset(root = './', split= 'val')
# test_set = queryPlanPGDataset(root = './', split= 'test')
