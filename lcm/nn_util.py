from tcnn import tcnn
from util.torch_util import genLayerSizes
import pytorch_lightning as pl
import torch
from torch import nn
from torch_geometric.utils import to_dense_batch,to_dense_adj
from util.custom_loss import aleatoric_loss
from torchmetrics.regression import SpearmanCorrCoef
from util.custom_loss import qErrorLossClass
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW

# create a function to generate the MLP layers
def produce_mlp(inputNumFeat,firstLayerSize,LastLayerSize,dropout,activation=nn.ReLU(), return_module=True):
    """This function generates a multi-layer perceptron with the specified number of layers and sizes.
    Args:
    inputNumFeat: int, the number of features in the input layer
    firstLayerSize: int, the number of neurons in the first layer
    LastLayerSize: int, the number of neurons in the last layer
    dropout: float, the dropout rate
    activation: torch.nn.Module, the activation function
    return_module: bool, whether to return the module or a list of layers
    Returns:
    nn.Module or list of nn.Module: the MLP model
    """
    mlp_layers = []
    mlp_layersizes = genLayerSizes(inputNumFeat,firstLayerSize,LastLayerSize)

    for l in mlp_layersizes:
        layerin,layerout = l[0],l[1]
        mlp_layers.append(nn.Linear(layerin,layerout))
        mlp_layers.append(nn.BatchNorm1d(layerout))
        mlp_layers.append(activation)
        mlp_layers.append(nn.Dropout(dropout))
    if return_module:
        return nn.Sequential(*mlp_layers)
    else:
        return mlp_layers

