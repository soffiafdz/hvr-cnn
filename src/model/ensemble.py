import torch
from torch import nn

class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        for idx, module in enumerate(models):
                self.add_module(str(idx), module)

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def forward(self, x):
        v = []

        for i in self:
            v.append( i.forward(x)['seg'].log_softmax(1) )
        
        # average outputs        
        x = torch.stack(v, 0).mean(dim=0,keepdim=False)
        
        # TODO: figure out if we need to transfer something else
        return {'seg':x}
    