# ------------------------ V0 -------------------------#
# Reproduce Lero
import numpy as np
from tcnn import tcnn
from itertools import combinations
import pytorch_lightning as pl
import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_geometric.loader import DataLoader
from torchmetrics.regression import SpearmanCorrCoef
from util.custom_loss import qErrorLossClass
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW

sigmoid = nn.Sigmoid()

def get_training_pair(candidates):
    if len(candidates) >= 2:
        X1, X2 = [], []

        i = 0
        while i < len(candidates) - 1:
            s1 = candidates[i]
            j = i + 1
            while j < len(candidates):
                s2 = candidates[j]
                X1.append(s1)
                X2.append(s2)
                j += 1
            i += 1
        return X1, X2
    if len(candidates) == 1:
        return [candidates[0]], [candidates[0]]

class LeroNet(nn.Module):
    def __init__(self, input_feature_dim, device) -> None:
        super(LeroNet, self).__init__()
        self.input_feature_dim = input_feature_dim
        self._cuda = False
        self.device = device

        self.tree_conv = nn.Sequential(
            tcnn.BinaryTreeConv(self.input_feature_dim, 256, device=self.device),
            tcnn.TreeLayerNorm(),
            tcnn.TreeActivation(nn.LeakyReLU()),
            tcnn.BinaryTreeConv(256, 128, device=self.device),
            tcnn.TreeLayerNorm(),
            tcnn.TreeActivation(nn.LeakyReLU()),
            tcnn.BinaryTreeConv(128, 64, device=self.device),
            tcnn.TreeLayerNorm(),
            tcnn.DynamicPooling(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, trees):
        return self.tree_conv(trees)


class LeroModelPairwise(pl.LightningModule):
    def __init__(self,
        dropout=0.1, 
        num_node = None,
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
        self.lero_model = LeroModel(
            dropout, 
            num_node,
            numPlanFeat,
            numPlanOrdFeat,
            numQueryGraphFeat,
            device,
            criterion,
            batch_size,
            lr,
            rlrop_patience,
            rlrop_factor,
        )
        self.criterion = criterion
        self.lr = lr
        self.rlrop_patience = rlrop_patience
        self.rlrop_factor = rlrop_factor
        self.batch_size = batch_size
        self.spearmans_corr = SpearmanCorrCoef()
        self.follow_batch = ['x_s']
        self.pair_batch_size = 120
    
    def get_pairs(self,batch):
        X1, X2 = get_training_pair(batch)
        return X1, X2
    
    def forward(self, batch):
        # Get raw pairs without creating DataLoaders
        X1, X2 = self.get_pairs(batch)
        
        # Create single DataLoader for all pairs
        combined_data = X1 + X2
        loader = DataLoader(
            combined_data, 
            batch_size=self.pair_batch_size,
            shuffle=False, 
            num_workers=0, 
            follow_batch=self.follow_batch
        )
        
        # Process all data in one pass
        all_preds = []
        all_labels = []
        for b in loader:
            pred = self.lero_model(b)
            all_preds.append(pred)
            all_labels.append(b.y)
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Split results back into X1 and X2
        half_len = len(X1)
        y_pred_1 = all_preds[:half_len,0].reshape(-1)
        y_pred_2 = all_preds[half_len:,0].reshape(-1)
        self.y_label_1 = all_labels[:half_len]
        self.y_label_2 = all_labels[half_len:]
        # 3. compute sigmoid(diff)
        diff = y_pred_1 - y_pred_2
        prob_y = sigmoid(diff)
        return prob_y

    def training_step(self, batch, batch_idx):
        y_pred = self(batch)
        labels = (self.y_label_1 > self.y_label_2).float()
        loss = self.criterion(y_pred, labels)

        scores = self.lero_model(batch)[:,0].reshape(-1).detach()
        labels=batch.y.detach()
        corr = self.spearmans_corr(scores, labels)

        metrics_dict = {
            'train_loss':loss, 
            'train_corr':corr, 
            }
        self.log_dict(
            metrics_dict, batch_size = self.batch_size,
            on_step=False, on_epoch=True
            )
        
        # Force cleanup
        torch.cuda.empty_cache()
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_pred = self(batch)
        labels = (self.y_label_1 > self.y_label_2).float()
        loss = self.criterion(y_pred, labels)

        scores = self.lero_model(batch)[:,0].reshape(-1).detach()
        labels=batch.y.detach()
        corr = self.spearmans_corr(scores, labels)
        
        metrics_dict = {
            'val_loss':loss, 
            'val_corr':corr,
            }
        self.log_dict(
            metrics_dict, batch_size = self.batch_size,
            on_step=False, on_epoch=True
            )
        
        # Force cleanup
        torch.cuda.empty_cache()
        
        return loss

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
        
class LeroModel(pl.LightningModule):
    def __init__(self,
                 dropout=0.1, 
                 num_node = None,
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
        self.numQueryGraphFeat = numQueryGraphFeat
        self.__device = device
        self.numPlanFeat = numPlanFeat
        self.numPlanOrdFeat = numPlanOrdFeat
        self.num_node = num_node
        self.criterion = criterion
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.rlrop_patience = rlrop_patience
        self.rlrop_factor = rlrop_factor
        self.spearmans_corr = SpearmanCorrCoef()
        self.qerror = qErrorLossClass()
        
        # Initialize the LeroNet: LeroNet takes two additional feature for cardinality and selectivity
        self.node_dim = 2
        self.lero_net = LeroNet(numPlanFeat[0]+self.node_dim-3, device)
    
    def forward(self, batch):
        x, edge_index, edge_attr, graph_attr = batch.x_s, batch.edge_index_s,batch.edge_attr_s, batch.graph_attr
        batch_index = batch.x_s_batch
        plan_attr, plan_ord = batch.plan_attr, batch.plan_ord
        edge_index = edge_index.long()
        plan_attr = plan_attr.reshape(-1, self.numPlanFeat[0], self.numPlanFeat[1]).float()[:,3:,:] # batch_size x plan_node_dim-3 x num_plan_nodes
        plan_ord = plan_ord.reshape(-1, self.numPlanOrdFeat[0], self.numPlanOrdFeat[1]).long()
        graph_attr = graph_attr.reshape(-1,self.numQueryGraphFeat)
        batch_size = batch.y.shape[0]

        x = x[:,:2]
        x_dense,_ = to_dense_batch(x,batch_index) # B x 5 x 6
        # extract the relevant tables only and extend the plan features with table features
        # get the table onehot encoding for all plan nodes 
        # plan_attr: B x 13 x 20
        table_onehot = plan_attr[:,-self.num_node:,:] # B x 5 x 24
        table_onehot = torch.transpose(table_onehot,1,2) # B x 24 x 5
        table_onehot = torch.repeat_interleave(table_onehot,self.node_dim,dim=1) # B x 144 x 5
    
        # more efficient way of appending the plan nodes with table features that is scalable in both batch size and node embedding size 
        filt_x = torch.transpose(x_dense,dim0=1,dim1=2) # B x 6 x 5
        filt_x = torch.tile(filt_x,dims=(1,self.numPlanFeat[1],1)) # B x 144 x 5
        filt_x = filt_x*table_onehot
        filt_x = torch.sum(filt_x,dim=-1) # B x 144
        filt_x = filt_x.reshape(batch_size,self.numPlanFeat[1],self.node_dim) # B x 24 x 6
        filt_x = torch.transpose(filt_x,dim0=1,dim1=2) # B x 6 x 24
        plan_attr = torch.concat((plan_attr,filt_x),1) # B x 19 x 24
        
        lero_in = (plan_attr, plan_ord)
        lero_out = self.lero_net(lero_in)

        sigma = torch.zeros(lero_out.shape).to(self.__device)
        return torch.stack((lero_out, sigma),dim=1)

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
        # return optimizer
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)