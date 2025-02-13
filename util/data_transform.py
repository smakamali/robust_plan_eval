import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.dataset import Dataset

#################  GENERAL PURPOSE FUNCTIONS  #################

def getStats(inputTensor):
    if inputTensor.dim() == 1:
        inputTensor = inputTensor.reshape(-1,1)
    mask_nan = inputTensor.isnan().any(axis=1)
    masked_tensor = inputTensor[~mask_nan]
    mean = masked_tensor.mean(axis=0)
    minimum = masked_tensor.min(axis=0)[0]
    maximum = masked_tensor.max(axis=0)[0]
    return mean, minimum, maximum

def minmaxScaling(X, minVal, maxVal):
    return (X - minVal)/(maxVal - minVal)

def minmaxScaling_inv(y, minVal, maxVal):
    return y*(maxVal - minVal) + minVal

def log10Transform(X):
    epsilon = 0.00001
    X = torch.log10(X+epsilon)
    return X

def exp10Transform(X):
    epsilon = 0.00001
    X = torch.pow(10,X) - epsilon
    return X

#################  INPUT TRANSFORMATIONS  #################

class nullImputation(BaseTransform):
    r"""replaces nan values with representative value provided by ref_data
    """
    def __init__(self, train_set):
        self.meanEdge, _, _ = getStats(train_set.edge_attr_s)
        self.meanNode, _, _ = getStats(train_set.x_s)
        # self.mean_graph, _, _ = getStats(train_set.graph_attr)

        # self.graph_attr_num =  int(train_set.graph_attr.shape[0]/len(train_set))
        # graph_attr_rsh = torch.reshape(train_set.graph_attr,(-1,self.graph_attr_num))
        # self.mean_graph, _, _ = getStats(graph_attr_rsh)

    def __call__(self, data: Dataset) -> Dataset:

        for j in range(data.x_s.shape[1]):
            data._data.x_s[:,j] = torch.nan_to_num(data.x_s[:,j],nan=self.meanNode[j])
        
        for j in range(data.edge_attr_s.shape[1]):
            data._data.edge_attr_s[:,j] = torch.nan_to_num(data.edge_attr_s[:,j],nan=self.meanEdge[j])

        # graph_attr = torch.reshape(data.graph_attr,(-1, self.graph_attr_num))
        # for j in range(graph_attr.shape[1]):
        #     graph_attr[:,j] = torch.nan_to_num(
        #         graph_attr[:,j],
        #         nan=self.mean_graph[j]
        #         )
        # data._data.graph_attr = graph_attr.reshape(-1)

        return data

class minmaxScale(BaseTransform):
    r"""replaces nan values with representative value provided by ref_data (train_set)
    """
    def __init__(self, train_set):
        _, minEdge, maxEdge = getStats(train_set.edge_attr_s)
        _, minNode, maxNode = getStats(train_set.x_s)

        self.graph_attr_num =  int(train_set.graph_attr.shape[0]/len(train_set))
        graph_attr_rsh = torch.reshape(train_set.graph_attr,(-1,self.graph_attr_num))
        _, min_graph, max_graph = getStats(graph_attr_rsh)

        self.nodeRange_s = (minNode, maxNode)
        self.edgeRange = (minEdge, maxEdge)
        self.graphRange = (min_graph, max_graph)

    def __call__(self, data: Dataset) -> Dataset:
        data._data.x_s = minmaxScaling(
            data.x_s, self.nodeRange_s[0], self.nodeRange_s[1]
            )

        data._data.edge_attr_s = minmaxScaling(
            data.edge_attr_s, self.edgeRange[0], self.edgeRange[1]
            )
        
        graph_attr = torch.reshape(data.graph_attr,(-1, self.graph_attr_num))
        data._data.graph_attr = minmaxScaling(
            graph_attr, self.graphRange[0], self.graphRange[1]
            )
        data._data.graph_attr = graph_attr.reshape(-1)

        return data

class dropConst(BaseTransform):
    def __init__(self, train_set):
        _, minEdge, maxEdge = getStats(train_set.edge_attr_s)
        self.edge_uniqueCols = (minEdge == maxEdge)
        _, minNode_s, maxNode_s = getStats(train_set.x_s)
        self.node_uniqueCols = (minNode_s == maxNode_s)

        # self.graph_attr_num =  int(train_set.graph_attr.shape[0]/len(train_set))
        # graph_attr_rsh = torch.reshape(train_set.graph_attr,(-1,self.graph_attr_num))
        # _, min_graph, max_graph = getStats(graph_attr_rsh)
        # self.graph_uniqueCols = (min_graph == max_graph)

    def __call__(self, data_set: Dataset) -> Dataset:
        data_set._data.edge_attr_s = data_set.edge_attr_s[:,~self.edge_uniqueCols]
        data_set._data.x_s = data_set.x_s[:,~self.node_uniqueCols]

        # print("before", data_set._data.graph_attr.shape)

        # graph_attr = torch.reshape(data_set.graph_attr,(-1, self.graph_attr_num))
        # graph_attr = graph_attr[:,~self.graph_uniqueCols]
        # graph_attr = graph_attr.reshape(-1)
        # data_set._data.graph_attr = graph_attr
        
        # print("after", data_set._data.graph_attr.shape)

        return data_set

#################  TARGET TRANSFORMATIONS  #################

class target_transform:
    r"""applis min-max scaling or its inverse on target values
    """
    def __init__(self, train_set, target = 'latency'): # fit
        
        self.target = target

        if self.target == 'latency':
            _, minTarget, maxTarget = getStats(train_set.y)       
        elif self.target == 'cost':
            _, minTarget, maxTarget = getStats(train_set.opt_cost)
        else:
            raise Exception('target must be either `latency` or `cost`')
        
        self.minTarget = minTarget
        self.maxTarget = maxTarget

    def transform(self, data_set: Dataset) -> Dataset:
        if self.target == 'latency':
            data_set._data.y_t = minmaxScaling(data_set.y, self.minTarget, self.maxTarget)
        elif self.target == 'cost':
            data_set._data.y_t = minmaxScaling(data_set.opt_cost, self.minTarget, self.maxTarget)
        else:
            raise Exception('target must be either `latency` or `cost`')
        return data_set
    
    def inverse_transform(self, y):
        out = minmaxScaling_inv(y, self.minTarget, self.maxTarget)
        return out
    
class target_log_transform:
    r"""applis min-max scaling or its inverse on target values
    """
    def __init__(self, train_set, target = 'latency'): # fit
        
        self.target = target
        
        if self.target == 'latency':
            y_t = log10Transform(train_set.y)
        elif self.target == 'cost':
            y_t = log10Transform(train_set.opt_cost)
        else:
            raise Exception('target must be either `latency` or `cost`')

        _, minTarget, maxTarget = getStats(y_t)       
        self.minTarget = minTarget
        self.maxTarget = maxTarget

    def transform(self, data_set: Dataset) -> Dataset:
        if self.target == 'latency':
            out = log10Transform(data_set.y)
        elif self.target == 'cost':
            out = log10Transform(data_set.opt_cost)
        else:
            raise Exception('target must be either `latency` or `cost`')
        
        data_set._data.y_t = minmaxScaling(out, self.minTarget, self.maxTarget)
        return data_set
    
    def inverse_transform(self, y):
        out = minmaxScaling_inv(y, self.minTarget, self.maxTarget)
        out = exp10Transform(out)
        return out
