import torch.nn as nn
import math


class FC(nn.Module):
    """
    Fully connected layer head
    Args:
        n_features (int): Number of features in the input.
        num_class (int): Number of class for classification.
        dropout_ratio (float): Dropout ratio to use Default: 0.2.
        batch_norm (bool): Whether to use batch norm or not. Default: ``False``.
    """
    def __init__(self, n_features, num_class, dropout_ratio=0.2, batch_norm=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(self.n_features)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        self.classifier = nn.Linear(n_features, num_class)
        nn.init.normal_(self.classifier.weight, 0, math.sqrt(2.0 / num_class))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape: (batch_size, n_features)
        
        returns:
            torch.Tensor: logits for classification.
        """

        x = self.dropout(x)
        if self.bn:
            x = self.bn(x)
        x = self.classifier(x)
        return x

class NParamFC(nn.Module):
    def __init__(self, n_features, num_class, params, dropout_ratio=0.2, batch_norm=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(self.n_features)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()

        self.classifier = nn.Linear(n_features, num_class)
        nn.init.normal_(self.classifier.weight, 0, math.sqrt(2.0 / num_class))

        self.param_clfs = {}
        for param, n in params.items():
            self.param_clfs[param] = nn.Linear(n_features, n)
            nn.init.normal_(self.param_clfs[param].weight, 0, math.sqrt(2.0 / n))
        self.param_clfs = nn.ModuleDict(self.param_clfs)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape: (batch_size, n_features)
        
        returns:
            torch.Tensor: logits for classification.
        """

        x = self.dropout(x)
        if self.bn:
            x = self.bn(x)

        x_sign = self.classifier(x)
        x_params = { param : clf(x) for param, clf in self.param_clfs.items() }

        return x_sign, x_params
