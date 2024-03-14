from typing import Optional, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.conv import TransformerConv 
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp

# this module will only support homogeneous graphs for now
# support for bipartied graphs will be added later if needed

class transformer_conv_v2(nn.Module):
    def __init__(self, 
        global_dim_in: int, 
        global_emb_dim: int, 
        global_dim_out: int,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        **kwargs,
         ):
        super(transformer_conv_v2,self).__init__()
        self.gnn_in_channels = in_channels + global_emb_dim
        if concat == True:
            self.gnn_out_channels = out_channels*heads
        else:
            self.gnn_out_channels = out_channels
        self.global_dim_in = global_dim_in
        self.global_emb_dim = global_emb_dim
        self.global_dim_out = global_dim_out
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_global = nn.Linear(self.global_dim_in,self.global_emb_dim)
        
        self.gnn = TransformerConv(
            in_channels = self.gnn_in_channels,
            out_channels = out_channels,
            heads = self.heads,
            dropout = self.dropout,
            edge_dim = self.edge_dim,
            concat = concat,
            **kwargs)

        self.gnn_norm = BatchNorm(self.gnn_out_channels)
        # *2 to support two types of pooling
        self.gnn_out = nn.Linear(self.gnn_out_channels*2+self.global_emb_dim,
                                self.global_dim_out)
                        
    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, # return_attention_weights=None,
                global_attr: OptTensor = None, batch_index: OptTensor = None):
        # get global embedding
        g = self.lin_global(global_attr)
        # broadcast global embeddings and concat to nodes
        gb = g[batch_index,:]
        # print("x",x[:25,:7])
        h = torch.concat((x,gb),1)
        # perform graph convolution, batch norm, and activation
        # print("batch_index",batch_index.shape)
        # print("h",h.shape)
        # print("edge_index",edge_index.shape)
        # print("edge_attr",edge_attr.shape)
        h = self.gnn(h, edge_index, edge_attr)
        h = self.gnn_norm(h)
        h = torch.relu(h)
        # print("x",x)
        # apply pooling to gather node info back to graph 
        xmp = gmp(h,batch_index)
        xap = gap(h,batch_index)
        # print("xmp",xmp[:5,:7])
        # print("xap",xap[:5,:7])
        p = torch.cat([xmp,xap], dim=1)
        # concat the node information to graph embedding
        # print("g",g)
        # print("p",p)
        
        g = torch.concat((p,g),1)
        # print("g",g.shape)
        # print(g)


        # feed graph representation through a linear layer to get the desired output dimensionality
        g = self.gnn_out(g)
        # print("g-out",g.shape)

        return g, h