import numpy as np
import torch.nn as nn
import pytorch_lightning as pl


def genLayerSizes(inputNumFeat = 12, firstLayerSize = 600, LastLayerSize = 30):
    if np.log2(firstLayerSize).is_integer():
        upperexp2 = firstLayerSize
    else:
        upperexp2 = (2**int(np.log2(firstLayerSize)))
    
    if np.log2(LastLayerSize).is_integer():
        lowerexp2 = LastLayerSize
    else:
        lowerexp2 = (2**int(np.log2(LastLayerSize)+1))
    # print('upperexp2',upperexp2)
    # print('lowerexp2',lowerexp2)
    layersizes = []
    layersizes.append(inputNumFeat)
    if firstLayerSize > upperexp2 and firstLayerSize != inputNumFeat:
        layersizes.append(firstLayerSize)
    nextvalue = upperexp2
    for i in range(100):
        if nextvalue >=lowerexp2 :
            if nextvalue != layersizes[-1]:
                layersizes.append(nextvalue)
            nextvalue=int(nextvalue/2)

        else:
            if LastLayerSize != lowerexp2:
                layersizes.append(LastLayerSize)
                break
    # print(layersizes)
    # layersizes = np.unique(np.array(layersizes))
    # print(layersizes)
    layers = []
    for idx, i in enumerate(layersizes):
        if idx <= len(layersizes)-2:
            layers.append([layersizes[idx], layersizes[idx+1]])
    # print(layers)
    return layers

# Enables variational inference using MC dropout
class LitMCdropoutModel(pl.LightningModule):
    def __init__(self, model, mc_iteration = 10, dropout = 0.1):
        super().__init__()
        self.model = model
        self.mc_iteration = mc_iteration
    def predict_step(self, batch, batch_idx):
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Dropout):
                layer.train(True)
        preds = [self.model(batch).squeeze().tolist() for _ in range(self.mc_iteration)]
        return preds