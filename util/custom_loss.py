import torch
import torch.nn as nn

class qErrorLossClass(nn.Module):
    # def __init__(self):
    #     super(qErrorLossClass, self).__init__()
    def forward(self, my_outputs, my_labels):
        eps = 0.0001
        denominator = torch.minimum(my_outputs,my_labels)
        numerator = torch.maximum(my_outputs,my_labels)
        qerror = numerator/(denominator+eps)
        return torch.mean(qerror)

class aleatoric_loss(nn.Module):
    def __init__(self, device=None):
        super(aleatoric_loss, self).__init__()
        self.device = device
    def forward(self, my_outputs, my_labels):
        # neg_log_likelihood = -my_outputs.log_prob(my_labels)
        # return torch.mean(neg_log_likelihood)
        eps = 0.0001
        mu = my_outputs[:,0,:]
        sigma = my_outputs[:,1,:]

        se = torch.pow((mu-my_labels),2)
        
        # print("se",se.shape)
        inv_std = torch.pow(sigma,-1)
        # print("inv_std",inv_std.shape)
        mse = (torch.mul(inv_std,se))
        
        reg = (torch.log(sigma+eps))
        # print("reg",torch.log(my_outputs[:,1,:]+eps).shape)
        cons = torch.log(torch.zeros(my_labels.shape)+(2*3.141592))
        if self.device is None:
            cons=cons.cpu()
        else:
            cons=cons.to(self.device)
        
        loss = (mse + reg + cons)*0.5
        return torch.mean(loss)

class rmse_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.functional.mse_loss
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat[:,0],y))