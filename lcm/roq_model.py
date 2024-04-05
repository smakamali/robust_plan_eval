from tcnn import tcnn
from util.torch_util import genLayerSizes
import pytorch_lightning as pl
import torch
from torch import nn
from gnn.query_graph_gnn import transformer_conv_v2
from torchmetrics.regression import SpearmanCorrCoef
from util.custom_loss import qErrorLossClass
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, AdamW

def produce_mlp(inputNumFeat,firstLayerSize,LastLayerSize,dropout,activation=nn.ReLU(), return_module=True):
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
                 with_var = False, device = None,
                 criterion = None,
                 batch_size = None,
                 lr = 0.001,
                 rlrop_patience = 10,
                 rlrop_factor = 0.5,
                 ):
        super().__init__()
        self.validation_step_outputs = []
        self.with_var = with_var
        self.ext_plan = ext_plan
        self.query_processor = query_processor
        self.numQueryGraphFeat = numQueryGraphFeat
        self.numPlanFeat = numPlanFeat
        self.numPlanOrdFeat = numPlanOrdFeat
        self.num_node = num_node
        self.node_dim = node_dim
        self.node_embd_dim = node_embd_dim
        self.qp_attheads = qp_attheads
        self.criterion = criterion
        self.batch_size = batch_size
        self.node_embd_dim_for_plan = node_embd_dim # used to compress node embeddings before appending to plan nodes - using node_embd_dim for now
        query_graph_embd_dim = query_module_out # for now
        self.lr = lr
        self.dropout = dropout
        self.rlrop_patience = rlrop_patience
        self.rlrop_factor = rlrop_factor
        self.spearmans_corr = SpearmanCorrCoef()
        self.qerror = qErrorLossClass()
        
        if ext_plan:
            self.tcnn_in_channels = numPlanFeat[0] + self.node_embd_dim_for_plan
        else:
            self.tcnn_in_channels = numPlanFeat[0]

        if query_processor == 'mlp':
            numQueryFeat = num_node*node_dim + num_edge*edge_dim + numQueryGraphFeat
            self.queryMLP = produce_mlp(
                inputNumFeat=numQueryFeat,
                firstLayerSize=query_module_in,
                LastLayerSize=query_module_out,
                dropout=dropout
                )
        elif query_processor == 'gnn':
            # GCNN module using TransformerConv

            qp_gnn_layers = []
            config = {'global_dim_in':numQueryGraphFeat, 
                    'global_emb_dim':query_graph_embd_dim,
                    # 'gnn_out_1st_layer': query_module_in,
                    'global_dim_out':query_module_out,
                    'in_channels':node_dim,
                    'out_channels':node_embd_dim,
                    'edge_dim':edge_dim,
                    'heads':qp_attheads,
                    'concat':True,
                    'dropout':dropout}
            
            for l in range(qp_gnn_ls):
                if l == 0: # first layer
                    qp_gnn_layers.append(transformer_conv_v2(**config))
                else: # hidden layers
                    config['global_dim_in']=query_module_out
                    config['in_channels']=node_embd_dim*qp_attheads
                    qp_gnn_layers.append(transformer_conv_v2(**config))
            self.qp_gnn_layers = nn.ModuleList(qp_gnn_layers)

            self.lin_low_node_emb=nn.Linear(node_embd_dim*qp_attheads, self.node_embd_dim_for_plan)

        else:
            raise Exception("query_processor must be either gnn or mlp.")

        # TCNN module from Neo
        self.guidelineTCNNLayers = []
        glTCNNLayerSizes = genLayerSizes(
            inputNumFeat = self.tcnn_in_channels, 
            firstLayerSize = TCNNin, 
            LastLayerSize = TCNNout
            )
        
        for l in glTCNNLayerSizes:
            layerin,layerout = l[0],l[1]
            self.guidelineTCNNLayers.append(tcnn.BinaryTreeConv(layerin, 
                                                                layerout, device=device))
            self.guidelineTCNNLayers.append(tcnn.TreeLayerNorm())
            self.guidelineTCNNLayers.append(tcnn.TreeActivation(nn.ReLU()))
#             self.guidelineTCNNLayers.append(nn.Dropout(dropout))
        self.guidelineTCNNLayers.append(tcnn.DynamicPooling())
        self.guidelineTCNN = nn.Sequential(*self.guidelineTCNNLayers)
        
        # Final layers
        self.final_layers_in_channels = TCNNout+query_module_out
        self.finalMLP = produce_mlp(
            inputNumFeat=self.final_layers_in_channels,
            firstLayerSize=finalMLPin,
            LastLayerSize=finalMLPout,
            dropout=dropout
            )
        
        # Mean parameters
        self.meanLayers = produce_mlp(
            inputNumFeat=finalMLPout,
            firstLayerSize=finalMLPout,
            LastLayerSize=32,
            dropout=dropout,
            return_module=False
            )
        
        self.meanLayers.append(nn.Linear(32, 1))
        self.meanLayers.append(nn.Sigmoid())
        self.mean_layer = nn.Sequential(*self.meanLayers)
        
        # Standard deviation parameters
        if with_var:
            self.stdLayers = produce_mlp(
                inputNumFeat=finalMLPout,
                firstLayerSize=finalMLPout,
                LastLayerSize=32,
                dropout=dropout,
                return_module=False
                )
            self.stdLayers.append(nn.Linear(32, 1))
            self.stdLayers.append(nn.Softplus())
            self.std_layer = nn.Sequential(*self.stdLayers)
    
    def forward(self, batch):
        x, edge_index, edge_attr, graph_attr = batch.x_s, batch.edge_index_s,batch.edge_attr_s, batch.graph_attr
        batch_index = batch.x_s_batch
        plan_attr, plan_ord = batch.plan_attr, batch.plan_ord
        edge_index = edge_index.long()
        plan_attr = plan_attr.reshape(-1, self.numPlanFeat[0], self.numPlanFeat[1]).float()
        plan_ord = plan_ord.reshape(-1, self.numPlanOrdFeat[0], self.numPlanOrdFeat[1]).long()
        graph_attr = graph_attr.reshape(-1,self.numQueryGraphFeat)
        batch_size = batch.y.shape[0]
        
        if self.query_processor == 'mlp':
            # flatten the whole graph into a one-dimensional vector
            x_rsh = x.reshape(batch_size,-1)
            edge_rsh = edge_attr.reshape(batch_size,-1)
            mlpin = torch.concat((x_rsh,edge_rsh,graph_attr),1)
            gcnn_out = self.queryMLP(mlpin)
        elif self.query_processor == 'gnn':
            # porcess node and edge attributes
            # hidden_embd = []
            # hidden_embd.append(graph_attr)
            for idx, gnn in enumerate(self.qp_gnn_layers):
                if idx == 0:
                    g,h = gnn (x, edge_index, edge_attr, graph_attr, batch_index)
                else:
                    g,h = gnn (h, edge_index, edge_attr, g, batch_index)
                # hidden_embd.append(h)
            h=self.lin_low_node_emb(h)
            gcnn_out = g

        else:
            raise Exception("query_processor must be either gnn or mlp.")
        
        # extract the relevant tables only and extend the plan features with table features
        if self.ext_plan:
            # get the table onehot encoding for all plan nodes 
            # plan_attr: B x 13 x 20
            table_onehot = plan_attr[:,-self.num_node:,:] # B x 5 x 20
            table_onehot = torch.transpose(table_onehot,1,2) # B x 20 x 5
            # print('table_onehot',table_onehot.shape)
            table_onehot = torch.repeat_interleave(table_onehot,self.node_embd_dim_for_plan,dim=1) # B x 120 x 5
            # table_onehot = table_onehot.reshape(-1,table_onehot.shape[2]) # B20 x 5
            # extract features of the relevant tables - ONLY COMPATIBLE WITH AUGMENTATION FOLD 0
            # x : B5 x 6
            if self.query_processor =='mlp':
                filt_x = batch.x.reshape(-1,self.num_node,self.node_dim) # B x 5 x 6
            else:
                filt_x = h.reshape(-1,self.num_node,self.node_embd_dim_for_plan) # B x 5 x 6
            
            # more efficient way of appending the plan nodes with table features that is scalable in both batch size and node embedding size 
            filt_x = torch.transpose(filt_x,dim0=1,dim1=2) # B x 6 x 5
            filt_x = torch.tile(filt_x,dims=(1,self.numPlanFeat[1],1)) # B x 120 x 5
            filt_x = torch.sum(filt_x*table_onehot,dim=-1) # B x 120
            filt_x = filt_x.reshape(batch_size,self.numPlanFeat[1],self.node_embd_dim_for_plan) # B x 20 x 6
            filt_x = torch.transpose(filt_x,dim0=1,dim1=2) # B x 6 x 20
            plan_attr = torch.concat((plan_attr,filt_x),1) # B x 19 x 20
        
        tcnn_in = (plan_attr, plan_ord)
        tcnn_out = self.guidelineTCNN(tcnn_in)
        
        final_layer_in = torch.concat((tcnn_out,gcnn_out),1)
        output_layers_in = self.finalMLP(final_layer_in)
        
        mu = self.mean_layer(output_layers_in)
        if self.with_var:
            sigma = self.std_layer(output_layers_in)
            return torch.stack((mu, sigma),dim=1)
        else:
            return mu
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        y_pred = self(batch)
        labels = batch.y_t.reshape(-1,1)
        loss = self.criterion(y_pred, labels)
        corr = self.spearmans_corr(y_pred[:,0], labels)
        qerr = self.qerror(y_pred[:,0], labels)
        metrics_dict = {'train_loss':loss, 'train_corr':corr, 'train_q-error':qerr}
        self.log_dict(
            metrics_dict, batch_size = self.batch_size,
            on_step=False, on_epoch=True
            )
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_pred = self(batch)
        labels = batch.y_t.reshape(-1,1)
        loss = self.criterion(y_pred, labels)
        corr = self.spearmans_corr(y_pred[:,0], labels)
        qerr = self.qerror(y_pred[:,0], labels)
        metrics_dict = {'val_loss':loss, 'val_corr':corr, 'val_q-error':qerr}
        self.log_dict(
            metrics_dict, batch_size = self.batch_size,
            on_step=False, on_epoch=True
            )
        return metrics_dict
    
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
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
