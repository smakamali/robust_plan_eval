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

def minmaxScaling(X, minVal, maxVal, target=False):
    """
    Performs min-max scaling while preserving constant features.
    
    Args:
        X: Input tensor to be scaled
        minVal: Tensor of minimum values for each feature
        maxVal: Tensor of maximum values for each feature
        
    Returns:
        Scaled tensor with constant features unchanged
    """
    if target:
        return (X - minVal)/(maxVal - minVal)
    
    else:
        # Create a mask for non-constant features (where max != min)
        range_mask = (maxVal - minVal) != 0
        
        # Initialize output tensor with copy of input
        X_scaled = X.clone()
        
        # Apply scaling only to non-constant features
        if range_mask.any():
            X_scaled[:, range_mask] = (X[:, range_mask] - minVal[range_mask]) / (maxVal[range_mask] - minVal[range_mask])
        
        return X_scaled

def minmaxScaling_inv(y, minVal, maxVal, target=False):
    """
    Performs inverse min-max scaling while preserving constant features.
    
    Args:
        y: Input tensor to be inverse scaled
        minVal: Tensor of minimum values for each feature
        maxVal: Tensor of maximum values for each feature
        
    Returns:
        Inverse scaled tensor with constant features unchanged
    """
    if target:
        return y*(maxVal - minVal) + minVal
    
    else:
        # Create a mask for non-constant features (where max != min)
        range_mask = (maxVal - minVal) != 0
        
        # Initialize output tensor with copy of input
        y_inv = y.clone()
        
        # Apply inverse scaling only to non-constant features
        if range_mask.any():
            y_inv[:, range_mask] = y[:, range_mask] * (maxVal[range_mask] - minVal[range_mask]) + minVal[range_mask]
        
        return y_inv

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
        node_attr_log = log10Transform(train_set.x_s)
        _, minNode, maxNode = getStats(node_attr_log)
    
        egde_attr_log = log10Transform(train_set.edge_attr_s)
        _, minEdge, maxEdge = getStats(egde_attr_log)

        self.graph_attr_num =  int(train_set.graph_attr.shape[0]/len(train_set))
        graph_attr_rsh = torch.reshape(train_set.graph_attr,(-1,self.graph_attr_num))
        graph_attr_log = log10Transform(graph_attr_rsh)
        _, min_graph, max_graph = getStats(graph_attr_log)

        plan_attr_shape = train_set[0].plan_attr.shape
        temp=train_set._data.plan_attr.clone()
        temp=temp.reshape(-1,plan_attr_shape[0],plan_attr_shape[1])
        temp=temp.transpose(1,2)
        plan_attr_rsh=temp.reshape(-1,plan_attr_shape[0])
        plan_attr_log = log10Transform(plan_attr_rsh)
        _, min_plan_attr, max_plan_attr = getStats(plan_attr_log)

        self.nodeRange_s = (minNode, maxNode)
        self.edgeRange = (minEdge, maxEdge)
        self.graphRange = (min_graph, max_graph)
        self.plan_attrRange = (min_plan_attr, max_plan_attr)

    def __call__(self, data: Dataset) -> Dataset:
        node_attr_log = log10Transform(data.x_s)
        data._data.x_s = minmaxScaling(
            node_attr_log, self.nodeRange_s[0], self.nodeRange_s[1]
            )

        egde_attr_log = log10Transform(data.edge_attr_s)
        data._data.edge_attr_s = minmaxScaling(
            egde_attr_log, self.edgeRange[0], self.edgeRange[1]
            )
        
        graph_attr = torch.reshape(data._data.graph_attr,(-1, self.graph_attr_num))
        graph_attr = minmaxScaling(
            graph_attr, self.graphRange[0], self.graphRange[1]
            )
        data._data.graph_attr = graph_attr.reshape(-1)

        # print("before:",data[0].plan_attr[3,:])
        # print("min:",self.plan_attrRange[0],"max:",self.plan_attrRange[1])
        plan_attr_shape = data[0].plan_attr.shape
        temp=data._data.plan_attr.clone()
        temp=temp.reshape(-1,plan_attr_shape[0],plan_attr_shape[1])
        temp=temp.transpose(1,2)
        temp=temp.reshape(-1,plan_attr_shape[0])
        plan_attr_log = log10Transform(temp)
        plan_attr_sc = minmaxScaling(
            plan_attr_log, self.plan_attrRange[0], self.plan_attrRange[1]
            )
        # Reshape the plan_attr_rsh back to original shape
        # 1. First reshape back to 3D tensor with the middle dimension transposed
        temp = plan_attr_sc.reshape(-1, plan_attr_shape[1], plan_attr_shape[0])
        # 2. Transpose back the middle and last dimensions
        temp = temp.transpose(1, 2)
        # 3. Finally reshape to match the original data._data.plan_attr shape
        data._data.plan_attr = temp.reshape(data._data.plan_attr.shape)
        # print("after:",data[0].plan_attr[3,:])

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
            data_set._data.y_t = minmaxScaling(data_set.y, self.minTarget, self.maxTarget, target=True)
        elif self.target == 'cost':
            data_set._data.y_t = minmaxScaling(data_set.opt_cost, self.minTarget, self.maxTarget, target=True)
        else:
            raise Exception('target must be either `latency` or `cost`')
        return data_set
    
    def inverse_transform(self, y):
        out = minmaxScaling_inv(y, self.minTarget, self.maxTarget, target=True)
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
        
        data_set._data.y_t = minmaxScaling(out, self.minTarget, self.maxTarget, target=True)
        return data_set
    
    def inverse_transform(self, y):
        out = minmaxScaling_inv(y, self.minTarget, self.maxTarget, target=True)
        out = exp10Transform(out)
        return out
