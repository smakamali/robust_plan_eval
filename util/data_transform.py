import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

def getStats(inputTensor):
    if inputTensor.dim() == 1:
        inputTensor = inputTensor.reshape(-1,1)
    mask_nan = inputTensor.isnan().any(axis=1)
    masked_tensor = inputTensor[~mask_nan]
    mean = masked_tensor.mean(axis=0)
    minimum = masked_tensor.min(axis=0)[0]
    maximum = masked_tensor.max(axis=0)[0]
    return mean, minimum, maximum

class NullImputation(BaseTransform):
    r"""replaces nan values with representative value provided by ref_data
    """
    def __init__(self, meanNode, meanEdge):
        self.meanNode = meanNode
        self.meanEdge = meanEdge

    def __call__(self, data: Data) -> Data:

        for j in range(data.x_s.shape[1]):
            data._data.x_s[:,j] = torch.nan_to_num(data.x_s[:,j],nan=self.meanNode[j])
        
        for j in range(data.edge_attr_s.shape[1]):
            data._data.edge_attr_s[:,j] = torch.nan_to_num(data.edge_attr_s[:,j],nan=self.meanEdge[j])

        return data

def minmaxScaling(X, minVal, maxVal):
    return (X - minVal)/(maxVal - minVal)

def minmaxScaling_inv(y, minVal, maxVal):
    return y*(maxVal - minVal) + minVal

### input transformation


class inputTransform(BaseTransform):
    r"""replaces nan values with representative value provided by ref_data
    """
    def __init__(self, nodeRange_s, edgeRange):
        self.nodeRange_s = nodeRange_s
        self.edgeRange = edgeRange

    def __call__(self, data: Data) -> Data:
        data._data.x_s = minmaxScaling(data.x_s, 
                               self.nodeRange_s[0], self.nodeRange_s[1])

        data._data.edge_attr_s = minmaxScaling(data.edge_attr_s, 
                                self.edgeRange[0], self.edgeRange[1])
        return data

def log10Transform(X):
    epsilon = 0.00001
    X = torch.log10(X+epsilon)
    return X

def exp10Transform(X):
    epsilon = 0.00001
    X = torch.pow(10,X) - epsilon
    return X

class target_transform:
    r"""applis min-max scaling or its inverse on target values
    """
    def __init__(self, y): # fit
        # y_t = log10Transform(y)
        _, minTarget, maxTarget = getStats(y)
        self.minTarget = minTarget
        self.maxTarget = maxTarget

    def transform(self, y):
        # out = log10Transform(y)
        out = minmaxScaling(y, self.minTarget, self.maxTarget)
        return out
    
    def inverse_transform(self, y):
        out = minmaxScaling_inv(y, self.minTarget, self.maxTarget)
        # out = exp10Transform(out)
        return out

class target_log_transform:
    r"""applis min-max scaling or its inverse on target values
    """
    def __init__(self, y): # fit
        y_t = log10Transform(y)
        _, minTarget, maxTarget = getStats(y_t)       
        self.minTarget = minTarget
        self.maxTarget = maxTarget

    def transform(self, y):
        out = log10Transform(y)        
        out = minmaxScaling(out, self.minTarget, self.maxTarget)
        return out
    
    def inverse_transform(self, y):
        out = minmaxScaling_inv(y, self.minTarget, self.maxTarget)
        out = exp10Transform(out)
        return out