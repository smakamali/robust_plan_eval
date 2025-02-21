# ----------------------------- V1 ----------------------------------#
# reproduce the Bao architecture, where the table cardinality is appended 
# to the tree nodes. This can be obtained by multiplying cardinality and
# selectivity (the first and the second join graph node features). This
# transformation is done inside the notebook before scaling the features.
# The first feature of x_s contains this feature
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
from lcm.nn_util import produce_mlp


class lcm_pl(pl.LightningModule):
    def __init__(self,
                 query_graph_embd_dim = 16,
                 query_module_in=256, # used only with query_processor = 'mlp'
                 query_module_out=32, 
                 node_embd_dim = 16,
                 TCNNin = 64, TCNNout = 32,
                 finalMLPin = 64, finalMLPout = 64, dropout=0.1, 
                 qp_attheads = 3,
                 qp_gnn_ls = 2,
                 ext_plan = True,
                 query_processor = 'gnn',
                 num_node = None,
                 node_dim = None,
                 num_edge = None, # used only with query_processor = 'mlp'
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
        self.query_processor = query_processor
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
        self.TCNNin = TCNNin
        self.TCNNout = TCNNout
        self.finalMLPin = finalMLPin
        self.finalMLPout = finalMLPout
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
        if 'neo' in self.architecture:
            self.node_dim = 2
            self.edge_dim = 3
            self.excluded_plan_node_features = 3
            self.queryMLP = self.build_query_MLP()
            self.tcnn_in_channels = self.numPlanFeat[0] + self.query_module_out - self.excluded_plan_node_features

            
        elif 'bao' in self.architecture:
            self.node_dim = 2
            # self.edge_dim = 0
            self.excluded_plan_node_features = 2
            self.tcnn_in_channels = self.numPlanFeat[0] + self.node_dim - self.excluded_plan_node_features

        
        else:
            raise Exception("architecture must be either `neo` or `bao`.")

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
        plan_attr = plan_attr.reshape(-1, self.numPlanFeat[0], self.numPlanFeat[1]).float()[:,self.excluded_plan_node_features:,:] # batch_size x plan_node_dim x num_plan_nodes
        plan_ord = plan_ord.reshape(-1, self.numPlanOrdFeat[0], self.numPlanOrdFeat[1]).long()
        graph_attr = graph_attr.reshape(-1,self.numQueryGraphFeat)
        batch_size = batch.y.shape[0]
        
        if 'neo' in self.architecture:
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
            qp_out = torch.repeat_interleave(qp_out,plan_attr.shape[2],dim=2) # B x 
            plan_attr = torch.concat((plan_attr,qp_out),dim=1)

        elif 'bao' in self.architecture:
            x = x[:,:self.node_dim]
            x_dense,_ = to_dense_batch(x,batch_index) # B x 5 x 6
            # extract the relevant tables only and extend the plan features with table features
            # get the table onehot encoding for all plan nodes 
            # plan_attr: B x 13 x 20
            table_onehot = plan_attr[:,-self.num_node:,:] # B x 5 x 24
            table_onehot = torch.transpose(table_onehot,1,2) # B x 24 x 5
            table_onehot = torch.repeat_interleave(table_onehot,self.node_dim,dim=1) # B x 144 x 5
        
            # more efficient way of appending the plan nodes with table features that is scalable in both batch size and node embedding size 
            filt_x = torch.transpose(x_dense,dim0=1,dim1=2) # B x 6 x 5
            filt_x = torch.tile(filt_x,dims=(1,plan_attr.shape[2],1)) # B x 144 x 5
            filt_x = filt_x*table_onehot
            filt_x = torch.sum(filt_x,dim=-1) # B x 144
            filt_x = filt_x.reshape(batch_size,plan_attr.shape[2],self.node_dim) # B x 24 x 6
            filt_x = torch.transpose(filt_x,dim0=1,dim1=2) # B x 6 x 24
            plan_attr = torch.concat((plan_attr,filt_x),1) # B x 19 x 24
        
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
