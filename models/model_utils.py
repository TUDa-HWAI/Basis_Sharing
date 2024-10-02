import torch
import torch.nn as nn
import torch.nn.functional as F


def build_basis_collection(groups, num_basis, nx):
    model_dict = torch.nn.ModuleDict()
    for group in groups:
        basis = Basis(num_basis, nx)
        for item in group:
            model_dict[str(item)] = basis
    return model_dict


class Basis(nn.Linear):
    def __init__(self, num_basis, nx):
        super().__init__(nx, num_basis, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def set_weight(self, weight):
        with torch.no_grad():
            self.weight.copy_(weight.T.detach().clone())


class Coefficient(nn.Linear):
    def __init__(self, nf, num_basis, bias=False):
        super().__init__(num_basis, nf, bias=bias)
        self.nf = nf

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = F.linear(x, self.weight, self.bias)
        x = x.view(size_out)
        return x

    def set_weight(self, weight, bias=None):
        with torch.no_grad():
            self.weight.copy_(weight.T.detach().clone())
