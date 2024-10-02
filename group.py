import torch
from tqdm import tqdm
from calib import Calib


class Group:
    def __init__(self, std_model, group_member, name, step, model_type, s, invs):
        """
        :param std_model: the original model
        :param group_member: the layers which share2 the same parameter
        :param names: list, share2 model name
        :param steps: list, the col num of each name
        """
        self.member = group_member
        self.model_type = model_type
        self.name = name
        self.step = step
        self.basis = None
        self.coefficient = None
        self.sigma = None
        self._init_basis_coefficient(std_model, s, invs)

    def _init_gpt2(self, std_model):
        assert self.model_type == 'gpt2'
        w = []
        model = std_model.transformer.h
        for layer in self.member:
            data = model[layer].get_submodule(self.name).weight.data
            w.append(data)
        return w

    def _init_llama2(self, std_model):
        assert self.model_type == 'llama2'
        w = []
        model = std_model.model.layers
        for layer in self.member:
            data = model[layer].get_submodule(self.name).weight.data
            w.append(data.T)
        return w

    def _init_opt(self, std_model):
        assert self.model_type == "opt"
        w = []
        model = std_model.model.decoder.layers
        for layer in self.member:
            data = model[layer].get_submodule(self.name).weight.data
            w.append(data.T)
        return w

    def _init_mistral(self, std_model):
        assert self.model_type == 'mistral'
        w = []
        model = std_model.model.layers
        for layer in self.member:
            data = model[layer].get_submodule(self.name).weight.data
            w.append(data.T)
        return w

    def _init_basis_coefficient(self, std_model, s, invs):
        if self.model_type == 'gpt2':
            w = self._init_gpt2(std_model)
        elif self.model_type == "llama2":
            w = self._init_llama2(std_model)
        elif self.model_type == "opt":
            w = self._init_opt(std_model)
        elif self.model_type == "mistral":
            w = self._init_mistral(std_model)
        else:
            raise NotImplementedError

        w = torch.cat(w, -1).double()
        s = s.to(w.device)
        invs = invs.to(w.device)
        w = s @ w
        u, sigma, v = torch.svd(w)
        # self.sigma = sigma
        self.basis = torch.matmul(invs @ u, torch.diag(sigma)).float()
        self.coefficient = v.T.float()

    def _get_coefficient_split(self):
        res = {}
        offset = self.step
        for i, layer in enumerate(self.member):
            res[layer] = {}
            start = offset * i
            co_attn = self.coefficient[:, start:start + offset]
            res[layer][self.name] = co_attn
        return res

    def change_basis(self, model, num_basis, basis_name):
        if self.model_type == 'gpt2':
            tmp_model = model.transformer
        elif self.model_type == "llama2":
            tmp_model = model.model
        elif self.model_type == "opt":
            tmp_model = model.model.decoder
        elif self.model_type == "mistral":
            tmp_model = model.model
        else:
            raise NotImplementedError
        tmp_model.get_submodule(basis_name)[str(self.member[0])].set_weight(self.basis[:, :num_basis])
        return model

    def change_coefficient(self, model, num_basis):
        if self.model_type == 'gpt2':
            tmp_model = model.transformer.h
        elif self.model_type == "llama2":
            tmp_model = model.model.layers
        elif self.model_type == "opt":
            tmp_model = model.model.decoder.layers
        elif self.model_type == "mistral":
            tmp_model = model.model.layers
        else:
            raise NotImplementedError
        co = self._get_coefficient_split()
        for i, layer in enumerate(self.member):
            weight = co[layer][self.name][:num_basis, :]
            tmp_model[layer].get_submodule(self.name).set_weight(weight)
        return model


def change_model(std_model, model, model_type, groups, name, step, num_basis, basis_name, calib_path):
    for group in tqdm(groups):
        s, inv_s = Calib.get_s_inv_s(group, name, model_type, calib_path)
        item = Group(std_model, group, name=name, step=step, model_type=model_type, s=s, invs=inv_s)
        model = item.change_basis(model, num_basis, basis_name)
        model = item.change_coefficient(model, num_basis)
    return model


def update_model(std_model, model, model_type, groups, name, step, num_basis, basis_name, calib_path):
    if model_type == "gpt2":
        tmp_std_model = std_model.transformer.h
        tmp_model = model.model.trtansformer.h
        tmp = model.model.transformer
    elif model_type == "llama2" or model_type == "mistral":
        tmp_std_model = std_model.model.layers
        tmp_model = model.model.layers
        tmp = model.model
    elif model_type == "opt":
        tmp_std_model = std_model.model.decoder.layers
        tmp_model = model.model.decoder.layers
        tmp = model.model.decoder
    else:
        raise NotImplementedError
    for group in tqdm(groups):
        w = []
        for layer_idx in group:
            if model_type == "gpt2":
                data = tmp_std_model[layer_idx].get_submodule(name).weight.data
            else:
                data = tmp_std_model[layer_idx].get_submodule(name).weight.data.T
            w.append(data)

        u = tmp.get_submodule(basis_name)[str(group[0])].weight.data.T
        u = u.double()
        assert u.shape[1] == num_basis

        w = torch.cat(w, -1).double().to(u.device)
        if basis_name == "q_basis" or basis_name == "v_basis":
            xtx = Calib.get_calib_data(group, "k_basis", calib_path)
        elif basis_name == "gate_basis":
            xtx = Calib.get_calib_data(group, "up_basis", calib_path)
        else:
            xtx = Calib.get_calib_data(group, basis_name, calib_path)
        xtx = xtx.to(u.device).double()

        inv = torch.inverse(u.T @ xtx @ u)
        vt = w.T @ xtx @ u @ inv
        v = vt.T

        for i, layer_idx in enumerate(group):
            data = v[:, i * step:(i + 1) * step]
            tmp_model[layer_idx].get_submodule(name).set_weight(data)

    return model
