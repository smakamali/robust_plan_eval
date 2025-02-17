# ----------------------------- V0 ----------------------------------#
# reproducing the model from the paper "Balsa: Learning a Query Optimizer Without Expert Demonstrations"
# TODO: 
# 1- implement the simulation phase by enumerating all subplans in each plan -> DONE!
# 2- implement the transition from simulation to the actual training phase -> DONE!
# 3- align the features of the query and plan nodes with the features used in the paper

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
from util.tcnn_util import enumerate_subplans

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

# create a new class that inherits from pl.LightningModule and implements the simulation phase where training is done by enumerating the subplans in each plan and training on them
class balsa_simulation(pl.LightningModule):
    def __init__(self,
                 query_graph_embd_dim = 16,
                 query_module_in=256,
                 query_module_out=32, 
                 node_embd_dim = 16,
                 TCNNin = 64, TCNNout = 32,
                 finalMLPin = 64, finalMLPout = 64, dropout=0.1, 
                 qp_attheads = 3,
                 qp_gnn_ls = 2,
                 ext_plan = True,
                 num_node = None,
                 node_dim = None,
                 num_edge = None,
                 edge_dim = None,
                 numPlanFeat=None,
                 numPlanOrdFeat=None,
                 numQueryGraphFeat = None,
                 device = None,
                 criterion = None,
                 batch_size = None,
                 architecture = 'neo',
                 lr = 0.001,
                 rlrop_patience = 10,
                 rlrop_factor = 0.5,
                 ):
        super().__init__()
        self.validation_step_outputs = []
        self.ext_plan = ext_plan
        self.numQueryGraphFeat = numQueryGraphFeat
        self.numPlanFeat = numPlanFeat
        self.numPlanOrdFeat = numPlanOrdFeat
        self.num_node = num_node
        self.num_edge = num_edge
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.node_embd_dim = node_embd_dim
        self.qp_attheads = qp_attheads
        self.__device = device
        self.criterion = criterion
        self.batch_size = batch_size
        self.architecture = architecture
        self.node_embd_dim_for_plan = node_embd_dim # used to compress node embeddings before appending to plan nodes - using node_embd_dim for now
        self.query_graph_embd_dim=query_graph_embd_dim
        self.query_module_in=query_module_in
        self.TCNNin=TCNNin
        self.TCNNout=TCNNout
        self.query_module_out = query_module_out # for now
        self.finalMLPin=finalMLPin
        self.finalMLPout=finalMLPout
        self.qp_gnn_ls=qp_gnn_ls

        self.lr = lr
        self.dropout = dropout
        self.rlrop_patience = rlrop_patience
        self.rlrop_factor = rlrop_factor
        self.spearmans_corr = SpearmanCorrCoef()
        self.qerror = qErrorLossClass()

        # set the number of features for the nodes and edges used in balsa
        self.node_dim = 2
        self.edge_dim = 3
        
        self.balsa_model = self.build_model()
    
    def build_model(self):
        return balsa_model(
            self.query_graph_embd_dim,
            self.query_module_in,
            self.query_module_out,
            self.node_embd_dim,
            self.TCNNin, self.TCNNout,
            self.finalMLPin, self.finalMLPout, self.dropout, 
            self.qp_attheads,
            self.qp_gnn_ls,
            self.ext_plan,
            self.num_node,
            self.node_dim,
            self.num_edge,
            self.edge_dim,
            self.numPlanFeat,
            self.numPlanOrdFeat,
            self.numQueryGraphFeat,
            self.__device,
            self.criterion,
            self.batch_size,
            self.lr,
            self.rlrop_patience,
            self.rlrop_factor)
    
    def process_graph_features(self, batch):
        """Extracts dense representations of graph features."""

        x, edge_index, edge_attr = batch.x_s, batch.edge_index_s.long(), batch.edge_attr_s
        batch_index = batch.x_s_batch

        x = x[:,:self.node_dim] # extract only the features used in Neo
        edge_attr = edge_attr[:,-self.edge_dim:] # extract only the features used in Neo
        x_dense,_ = to_dense_batch(x,batch_index,self.num_node) # B x num_node x node_dim
        x_rsh = x_dense.reshape(self.batch_size,-1) # B x node_dim*num_node
        adj_dense = to_dense_adj(edge_index,batch_index,edge_attr=edge_attr,max_num_nodes=self.num_node) # B x num_node x num_node x edge_dim
        edge_rsh = adj_dense.reshape(self.batch_size,-1)

        return x_rsh, edge_rsh
    
    def process_plan_features(self, batch, qp_out_batch):
        
        plan_attr, plan_ord = batch.plan_attr, batch.plan_ord
        
        plan_attr = plan_attr.reshape(-1, self.numPlanFeat[0], self.numPlanFeat[1]).float() # B x numPlanFeat[0] x numPlanFeat[1] => B x plan_node_dim x num_plan_nodes
        plan_ord = plan_ord.reshape(-1, self.numPlanOrdFeat[0], self.numPlanOrdFeat[1]).long() # B x 3*(num_plan_nodes-1) x 1
        subplan_trees, subplan_indexes = enumerate_subplans(plan_attr,plan_ord)
        
        plan_attributes = []
        plan_indexes = []
        plan_costs = []
        for i in range(self.batch_size):
            subplan_attr = subplan_trees[i]
            subplan_ord = subplan_indexes[i]
            subplan_cost = subplan_attr[:,2].max(dim=-1)[0] # num_subplans
            qp_out = qp_out_batch[i,:].unsqueeze(0).unsqueeze(-1) # 1 x query_module_out x 1
            qp_out = torch.repeat_interleave(qp_out,subplan_attr.shape[0],dim=0) # num_subplans x query_module_out x 1
            qp_out = torch.repeat_interleave(qp_out,subplan_attr.shape[2],dim=2) # num_subplans x query_module_out x num_plan_nodes 
            subplan_attr = torch.concat((subplan_attr,qp_out),dim=1) # num_subplans x (plan_node_dim+query_module_out) x num_plan_nodes
            plan_attributes.append(subplan_attr)
            plan_indexes.append(subplan_ord)
            plan_costs.append(subplan_cost)
        plan_attributes = torch.concat(plan_attributes,dim=0).to(self.__device).float()
        plan_indexes = torch.concat(plan_indexes,dim=0).to(self.__device).long()
        plan_costs = torch.concat(plan_costs,dim=0).to(self.__device).float().reshape(-1,1)

        return plan_attributes, plan_indexes, plan_costs

    def forward(self, batch):        
        self.batch_size = batch.y.shape[0]

        x_rsh, edge_rsh = self.process_graph_features(batch)
        
        mlpin = torch.concat((x_rsh,edge_rsh),1)
        qp_out_batch = self.balsa_model.queryMLP(mlpin) # B x query_module_out
        
        plan_attributes, plan_indexes, plan_costs = self.process_plan_features(batch, qp_out_batch)

        tcnn_in = (plan_attributes, plan_indexes)
        tcnn_out = self.balsa_model.guidelineTCNN(tcnn_in)

        output_layers_in = self.balsa_model.finalMLP(tcnn_out)
        mu = self.balsa_model.mean_layer(output_layers_in)
        sigma = self.balsa_model.std_layer(output_layers_in) if isinstance(self.criterion, aleatoric_loss) else torch.zeros_like(mu).to(self.__device)
        
        return torch.stack((mu, sigma),dim=1), plan_costs
    
    def step(self, batch, stage="train"):
        """Computes loss and logs metrics for training/validation."""
        y_pred, labels = self(batch)
        loss = self.criterion(y_pred, labels)

        metrics_dict = {
            f"{stage}_loss": loss,
            f"{stage}_corr": self.spearmans_corr(y_pred[:, 0], labels),
            f"{stage}_q-error": self.qerror(y_pred[:, 0], labels)
        }
        self.log_dict(metrics_dict, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return metrics_dict
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, stage="train")["train_loss"]
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, stage="val")
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,patience=self.rlrop_patience,factor=self.rlrop_factor)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                }
            }


class balsa_model(pl.LightningModule):
    def __init__(self,
                 query_graph_embd_dim = 16,
                 query_module_in=256,
                 query_module_out=32, 
                 node_embd_dim = 16,
                 TCNNin = 64, TCNNout = 32,
                 finalMLPin = 64, finalMLPout = 64, dropout=0.1, 
                 qp_attheads = 3,
                 qp_gnn_ls = 2,
                 ext_plan = True,
                 num_node = None,
                 node_dim = None,
                 num_edge = None,
                 edge_dim = None,
                 numPlanFeat=None,
                 numPlanOrdFeat=None,
                 numQueryGraphFeat = None,
                 device = None,
                 criterion = None,
                 batch_size = None,
                 lr = 0.001,
                 rlrop_patience = 10,
                 rlrop_factor = 0.5,
                 ):
        super().__init__()
        self.validation_step_outputs = []
        self.ext_plan = ext_plan
        self.numQueryGraphFeat = numQueryGraphFeat
        self.numPlanFeat = numPlanFeat
        self.numPlanOrdFeat = numPlanOrdFeat
        self.num_node = num_node
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.node_embd_dim = node_embd_dim
        self.TCNNin = TCNNin
        self.TCNNout = TCNNout
        self.finalMLPin = finalMLPin
        self.finalMLPout = finalMLPout
        self.qp_attheads = qp_attheads
        self.__device = device
        self.criterion = criterion
        self.batch_size = batch_size
        self.node_embd_dim_for_plan = node_embd_dim # used to compress node embeddings before appending to plan nodes - using node_embd_dim for now
        self.query_module_in = query_module_in
        self.query_module_out = query_module_out # for now
        self.lr = lr
        self.dropout = dropout
        self.rlrop_patience = rlrop_patience
        self.rlrop_factor = rlrop_factor
        self.spearmans_corr = SpearmanCorrCoef()
        self.qerror = qErrorLossClass()
        
        # set parameters based on `architecture`
            
        self.node_dim = 2
        self.edge_dim = 3
        
        self.queryMLP = self.build_query_MLP()
        self.guidelineTCNN = self.build_tcnn()
        self.finalMLP, self.mean_layer, self.std_layer = self.build_final_layers()
        
    def build_query_MLP(self):
        numQueryFeat = self.num_node*self.node_dim + self.num_node*self.num_node*self.edge_dim
        return produce_mlp(
            inputNumFeat=numQueryFeat,
            firstLayerSize=self.query_module_in,
            LastLayerSize=self.query_module_out,
            dropout=self.dropout
            )
    
    def build_tcnn(self):
        self.tcnn_in_channels = self.numPlanFeat[0] + self.query_module_out
        
        # TCNN module from Neo and Bao
        self.guidelineTCNNLayers = []
        glTCNNLayerSizes = genLayerSizes(
            inputNumFeat = self.tcnn_in_channels, 
            firstLayerSize = self.TCNNin, 
            LastLayerSize = self.TCNNout
        )
        for l in glTCNNLayerSizes:
            layerin,layerout = l[0],l[1]
            self.guidelineTCNNLayers.append(tcnn.BinaryTreeConv(layerin, layerout, device=self.__device))
            self.guidelineTCNNLayers.append(tcnn.TreeLayerNorm())
            self.guidelineTCNNLayers.append(tcnn.TreeActivation(nn.ReLU()))

        self.guidelineTCNNLayers.append(tcnn.DynamicPooling())
        return nn.Sequential(*self.guidelineTCNNLayers)
    
    def build_final_layers(self):
        # Final layers
        self.final_layers_in_channels = self.TCNNout
        finalMLP = produce_mlp(
            inputNumFeat=self.final_layers_in_channels,
            firstLayerSize=self.finalMLPin,
            LastLayerSize=self.finalMLPout,
            dropout=self.dropout
        )
        
        # Mean parameters
        meanLayers = produce_mlp(
            inputNumFeat=self.finalMLPout,
            firstLayerSize=self.finalMLPout,
            LastLayerSize=32,
            dropout=self.dropout,
            return_module=False
        )
        meanLayers.append(nn.Linear(32, 1))
        meanLayers.append(nn.Sigmoid())
        mean_layer = nn.Sequential(*meanLayers)
        
        # Standard deviation parameters
        std_layer = None
        if isinstance(self.criterion,aleatoric_loss):
            stdLayers = produce_mlp(
                inputNumFeat=self.finalMLPout,
                firstLayerSize=self.finalMLPout,
                LastLayerSize=32,
                dropout=self.dropout,
                return_module=False
            )
            stdLayers.append(nn.Linear(32, 1))
            stdLayers.append(nn.Softplus())
            std_layer = nn.Sequential(*stdLayers)
        return finalMLP, mean_layer, std_layer

    def forward(self, batch):
        x, edge_index, edge_attr, graph_attr = batch.x_s, batch.edge_index_s,batch.edge_attr_s, batch.graph_attr
        batch_index = batch.x_s_batch
        plan_attr, plan_ord = batch.plan_attr, batch.plan_ord
        edge_index = edge_index.long()
        plan_attr = plan_attr.reshape(-1, self.numPlanFeat[0], self.numPlanFeat[1]).float()
        plan_ord = plan_ord.reshape(-1, self.numPlanOrdFeat[0], self.numPlanOrdFeat[1]).long()
        graph_attr = graph_attr.reshape(-1,self.numQueryGraphFeat)
        batch_size = batch.y.shape[0]
        
        x = x[:,:self.node_dim] # extract only the features used in Neo
        edge_attr = edge_attr[:,-self.edge_dim:] # extract only the features used in Neo

        x_dense,_ = to_dense_batch(x,batch_index,self.num_node) # B x 5 x 6
        x_rsh = x_dense.reshape(batch_size,-1) # B x 30
        adj_dense = to_dense_adj(edge_index,batch_index,edge_attr=edge_attr,max_num_nodes=self.num_node) # B x N x N x len(edge_attr)
        edge_rsh = adj_dense.reshape(batch_size,-1)
        
        mlpin = torch.concat((x_rsh,edge_rsh),1)

        # flatten the whole graph into a one-dimensional vector
        qp_out = self.queryMLP(mlpin) # B x query_module_out
        qp_out = torch.unsqueeze(qp_out,dim=2)
        qp_out = torch.repeat_interleave(qp_out,self.numPlanFeat[1],dim=2) # B x 
        plan_attr = torch.concat((plan_attr,qp_out),dim=1)
        tcnn_in = (plan_attr, plan_ord)
        tcnn_out = self.guidelineTCNN(tcnn_in)
        
        final_layer_in = tcnn_out
        output_layers_in = self.finalMLP(final_layer_in)
        
        mu = self.mean_layer(output_layers_in)
        if isinstance(self.criterion,aleatoric_loss):
            sigma = self.std_layer(output_layers_in)
        else:
            sigma = torch.zeros(mu.shape).to(self.__device)
        return torch.stack((mu, sigma),dim=1)
    
    def step(self, batch, stage="train"):

        y_pred = self(batch)
        labels = batch.y_t.reshape(-1,1)
        loss = self.criterion(y_pred, labels)
        corr = self.spearmans_corr(y_pred[:,0], labels)
        qerr = self.qerror(y_pred[:,0], labels)
        metrics_dict = {f"{stage}_loss":loss, f"{stage}_corr":corr, f"{stage}_q-error":qerr}
        self.log_dict(
            metrics_dict, batch_size = self.batch_size,
            on_step=False, on_epoch=True
            )
        return metrics_dict
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, stage="train")["train_loss"]
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, stage="val")
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer,patience=self.rlrop_patience,factor=self.rlrop_factor)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                }
            }
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
