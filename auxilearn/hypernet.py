from abc import abstractmethod

from torch import nn
from torch.nn.utils import weight_norm
import torch
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class HyperNet(nn.Module):
    def __init__(self, main_task, input_dim):
        super().__init__()
        self.main_task = main_task
        self.input_dim = input_dim

    def forward(self, losses, outputs=None, labels=None, data=None):
        pass

    def _init_weights(self):
        pass

    def get_weights(self):
        return list(self.parameters())


class MonoHyperNet(HyperNet):
    """Monotonic Hypernets

    """
    def __init__(self, main_task, input_dim, clamp_bias=False):
        super().__init__(main_task=main_task, input_dim=input_dim)
        self.clamp_bias = clamp_bias

    def get_weights(self):
        return list(self.parameters())

    @abstractmethod
    def clamp(self):
        pass

class MonoNonlinearHyperNet(MonoHyperNet):

    def __init__(
        self,
        main_task,
        input_dim,
        hidden_sizes=1,
        nonlinearity=None,
        bias=True,
        dropout_rate=0.,
        init_upper=None,
        init_lower=None,
        weight_normalization=True
    ):
        super().__init__(main_task=main_task, input_dim=input_dim)

        assert isinstance(hidden_sizes, (list, int)), "hidden sizes must be int or list"
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        self.nonlinearity = nonlinearity if nonlinearity is not None else nn.Softplus()
        self.dropout = nn.Dropout(dropout_rate)
        self.weight_normalization = weight_normalization

        if isinstance(self.nonlinearity, Identity):
            bias = False

        self.bias = bias
        dims = [self.input_dim] + hidden_sizes + [1]
        self.layers = []

        for j in range(len(dims) - 2):
            self.layers.append(
                self._get_layer(dims[j], dims[j + 1], init_upper=init_upper, init_lower=init_lower, bias=bias)
            )
            self.layers.append(self.nonlinearity)
            self.layers.append(self.dropout)

        self.layers.append(
            self._get_layer(dims[-2], dims[-1], init_upper=init_upper, init_lower=init_lower, bias=False)
        )

        self.net = nn.Sequential(*self.layers)

    def _get_layer(self, input_dim, output_dim, init_upper, init_lower, bias):

        layer = nn.Linear(input_dim, output_dim, bias=bias)
        self._init_layer(layer, init_upper=init_upper, init_lower=init_lower)
        if self.weight_normalization:
            return weight_norm(layer)
        return layer

    @staticmethod
    def _init_layer(layer, init_lower, init_upper):
        b = init_upper if init_upper is not None else 1.
        a = init_lower if init_lower is not None else 0.
        if isinstance(layer, nn.Linear):
            layer.weight = nn.init.uniform_(layer.weight, b=b, a=a)
            if layer.bias is not None:
                layer.bias = nn.init.constant_(layer.bias, 0.)

    def forward(self, losses, outputs=None, labels=None, data=None):
        main_loss = losses[:, self.main_task].mean()
        return self.net(losses).mean() + main_loss

    def clamp(self):
        for l in self.net:
            if isinstance(l, nn.Linear):
                if self.weight_normalization:
                    l.weight_v.data.clamp_(0)
                    l.weight_g.data.clamp_(0)
                else:
                    l.weight.data.clamp_(0)

                if l.bias is not None and self.clamp_bias:
                    l.bias.data.clamp_(0)

class MonoJoint(MonoHyperNet):

    def __init__(self,main_task,input_dim,feature_dim,hidden_sizes=1,nonlinearity=None,bias=True,dropout_rate=0.,weight_normalization=True,K = 2,init_lower= 0.0, init_upper=1.0):
        super().__init__(main_task=main_task, input_dim=input_dim)

        assert isinstance(hidden_sizes, (list, int)), "hidden sizes must be int or list"
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        self.nonlinearity = nonlinearity if nonlinearity is not None else nn.Softplus()
        self.dropout = nn.Dropout(dropout_rate)
        self.weight_normalization = weight_normalization
        self.task_prior = nn.Parameter(torch.ones(1,input_dim))
        self.feature_sel = nn.Linear(feature_dim, input_dim)
        self._init_layer(self.feature_sel, init_upper=0.05, init_lower=-0.05) 
        self.loss_w = nn.Parameter( 0.05*torch.rand(1,input_dim) )
        self.loss_b = nn.Parameter( torch.zeros(1,input_dim) )
        self.layers = []
        self.net = nn.Sequential(*self.layers)
    
    def norm_loss1(self,losses):
        ###### this performs bad


        m = losses.mean(dim=0,keepdim = True)
        std = losses.std( 0, keepdim=True)
        return (losses - m) / (std + 1e-6)

    @staticmethod
    def _init_layer(layer, init_lower, init_upper):
        b = init_upper if init_upper is not None else 1.
        a = init_lower if init_lower is not None else 0.
        if isinstance(layer, nn.Linear):
            layer.weight = nn.init.uniform_(layer.weight, b=b, a=a)
            if layer.bias is not None:
                layer.bias = nn.init.constant_(layer.bias, 0.)
    
    def get_loss_weight(self, losses):
        out = F.sigmoid(losses*self.loss_w + self.loss_b)
        return out


    def forward(self, features, losses, to_train = True):
        detached_losses = losses.detach()
        detached_features = features.detach()
        detached_losses = self.norm_loss1(detached_losses)

        feature_mask = self.feature_sel(detached_features)     
        label_mask = self.get_loss_weight(detached_losses)

        data_mask =  self.nonlinearity(feature_mask)*label_mask*self.nonlinearity(self.task_prior)
        if not to_train:
            return ( (data_mask*losses).sum(1) ).mean()
        else:
            detached_mask = data_mask.detach()
            return ((detached_mask*losses).sum(1)).mean() 


    def clamp(self):
        for l in self.net:
            if isinstance(l, nn.Linear):
                if self.weight_normalization:
                    l.weight_v.data.clamp_(0)
                    l.weight_g.data.clamp_(0)
                else:
                    l.weight.data.clamp_(0)

                if l.bias is not None and self.clamp_bias:
                    l.bias.data.clamp_(0)

