import os
import random

from matplotlib import pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np
import math
from Requirements.learner_test import Learner
from copy import deepcopy
import argparse
# from tensorboardX import SummaryWriter
from datetime import datetime
from sklearn.metrics import roc_auc_score
import traceback    

class ReadHead(nn.Module):
    """
    this is the readhead class of PanPep

    Parameters:
        param memory: a memory block used for retrieving the memory

    Returns:
        the similarity weights based on the memory basis, output by the forward function
    """

    def __init__(self, memory):
        super(ReadHead, self).__init__()
        self.memory = memory

    def forward(self, peptide):
        q = self.memory.Query(peptide)
        w = self.memory(q)
        return w


class WriteHead(nn.Module):
    """
    this is the writehead class of PanPep

    Parameters:
        param memory: a memory block used for retrieving the memory
        param C: the number of basis

    Returns:
        the forward function of this class is used to write the model into the memory block
    """

    def __init__(self, C, memory):
        super(WriteHead, self).__init__()
        self.memory = memory
        self.C = C
        # linear layer for transforming the past models into the memory
        self.model_transform = nn.Linear(208, self.C)
        nn.init.xavier_uniform_(self.model_transform.weight, gain=1.4)
        nn.init.normal_(self.model_transform.bias, std=0.01)

    def forward(self, thetas):
        with torch.no_grad():
            models = thetas.T
        w = self.model_transform(models)
        self.memory.writehead(w)


# Memory
class Memory(nn.Module):
    """
    this is the writehead class of PanPep

    Parameters:
        param memory: a memory block used for retrieving the memory
        param R: the length of identity matrix
        param L: the length of peptide embedding
        param C: the number of basis
        param V: the length of model parameter vector
        param num_task_batch : the number of tasks in one batch

    Returns:
        the task-level similarity based on the basis matrix in the memory block, output by the forward function
    """

    def __init__(self, L, C, R, V, num_task_batch=1):
        super(Memory, self).__init__()
        self.C = C
        self.R = R
        self.V = V
        self.num_task_batch = num_task_batch

        # the content memory matrix
        self.initial_state = torch.ones(C, V) * 1e-6
        self.register_buffer("content_memory", self.initial_state.data)

        # the basis matrix
        self.diognal = torch.eye(C)
        self.register_buffer("peptide_index", self.diognal.data)

        # the query matrix
        self.Query = nn.Linear(L, R)
        nn.init.xavier_uniform_(self.Query.weight, gain=1.4)
        nn.init.normal_(self.Query.bias, std=0.01)

    def forward(self, query):
        query = query.view(self.num_task_batch, 1, -1)
        w = F.softmax(F.cosine_similarity(self.peptide_index + 1e-16, query + 1e-16, dim=-1), dim=1)
        return w

    def reset(self):
        if torch.cuda.is_available():
            self.content_memory.data = self.initial_state.data.cuda()
        else:
            self.content_memory.data = self.initial_state.data.cpu()

    def size(self):
        return self.C, self.R, self.V

    def readhead(self, w):
        return torch.matmul(w.unsqueeze(1), self.content_memory).squeeze(1)

    def writehead(self, w):
        self.content_memory = w.T


# memory based net parameter reconstruction
def _split_parameters(x, memory_parameters):
    """
    This function is used to rebuild the model parameter shape from the parameter vector

    Parameters:
        param x: parameter vector
        param memory_parameters: origin model parameter shape

    Returns:
        a new model parameter shape from the parameter vector
    """

    new_weights = []
    start_index = 0
    for i in range(len(memory_parameters)):
        end_index = np.prod(memory_parameters[i].shape)
        new_weights.append(x[:, start_index:start_index + end_index].reshape(memory_parameters[i].shape))
        start_index += end_index
    return new_weights


class Memory_module(nn.Module):
    """
    this is the Memory_module class of PanPep

    Parameters:
        param memory: the memory block object
        param readhead: the read head object
        param writehead: the write head object
        param prev_loss: store previous loss for disentanglement distillation
        param prev_data: store previous data for disentanglement distillation
        param models: store previous models for disentanglement distillation
        param optim: This is the optimizer for the disentanglement distillation
    """

    def __init__(self, args, params_num):
        super(Memory_module, self).__init__()
        self.memory = Memory(args.L, args.C, args.R, params_num, num_task_batch=1)
        self.readhead = ReadHead(self.memory)
        self.writehead = WriteHead(args.C, self.memory)
        self.prev_loss = []
        self.prev_data = []
        if torch.cuda.is_available():
            self.models = torch.Tensor().cuda()
        else:
            self.models = torch.Tensor().cpu()
        self.optim = optim.Adam(self.parameters(), lr=5e-4)

    def forward(self, index):
        r = self.readhead(index)
        return r

    def reset(self):
        # reset the stored elements in the memory
        self.memory.reset()
        self.prev_data = []
        self.prev_loss = []
        if torch.cuda.is_available():
            self.models = torch.Tensor().cuda()
        else:
            self.models = torch.Tensor().cpu()

    def reinitialization(self):
        # the memory module parameter reinitialization
        nn.init.xavier_uniform_(self.memory.Query.weight, gain=1.4)
        nn.init.normal_(self.memory.Query.bias, std=0.01)
        nn.init.xavier_uniform_(self.writehead.model_transform.weight, gain=1.4)
        nn.init.normal_(self.writehead.model_transform.bias, std=0.01)


class Memory_Meta(nn.Module):
    """
    Meta Learner

    Parameters:
        param update_lr: the update learning rate
        param update_step_test: update steps
        param net: the model from the config parameters
        param meta_Parameter_nums: the number of model parameter
        param Memory_module: the Memory_module block
        param prev_loss: store previous loss for disentanglement distillation
        param prev_data: store previous data for disentanglement distillation
        param models: store previous models for disentanglement distillation
    """

    def __init__(self, args, config):
        super(Memory_Meta, self).__init__()

        # Set the updating parameter
        self.update_lr = args.update_lr
        self.update_step_test = args.update_step_test
        self.net = Learner(config)

        # Count the number of parameters
        tmp = filter(lambda x: x.requires_grad, self.net.parameters())
        self.meta_Parameter_nums = sum(map(lambda x: np.prod(x.shape), tmp))
        self.Memory_module = None

        # self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.prev_loss = []
        self.prev_data = []
        if torch.cuda.is_available():
            self.models = torch.Tensor().cuda()
        else:
            self.models = torch.Tensor().cpu()

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        this is the function for in-place gradient clipping.

        Parameters:
            param grad: list of gradients
            param max_norm: maximum norm allowable
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    def reset(self):
        self.prev_data = []
        self.prev_loss = []
        if torch.cuda.is_available():
            self.models = torch.Tensor().cuda()
        else:
            self.models = torch.Tensor().cpu()

    def get_embedding(self, x_spt, y_spt, x_qry):
        net = deepcopy(self.net)
        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters(), retain_graph=True)
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        if self.update_step_test >= 1:
            for k in range(1, self.update_step_test):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = net(x_spt, fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt)
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
        # predict logits
        embedding = net(x_qry, fast_weights, bn_training=False, return_embedding=True).cpu().detach().numpy()
        return embedding

    def finetunning(self, peptide, x_spt, y_spt, x_qry, balance_loss=False, return_params=False):
        """
        this is the function used for fine-tuning on support set and test on the query set

        Parameters:
            param peptide: the embedding of peptide
            param x_spt: the embedding of support set
            param y_spt: the labels of support set
            param x_qry: the embedding of query set

        Return:
            the binding scores for the TCRs in the query set in the few-shot setting
        """
        print(f"x_qry shape: {x_qry.shape}")  # 输入数据的维度
        print(f"x_qry size: {x_qry.size(0)}")  # batch size
        querysz = x_qry.size(0)
        print(f"querysz: {querysz}")
        start = []
        end = []

        # in order not to ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        if balance_loss:
            loss = F.cross_entropy(logits, y_spt, weight=torch.tensor([2, 1], device=y_spt.device, dtype=torch.float))
        else:
            loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters(), retain_graph=True)
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # the loss and accuracy before first update
        with torch.no_grad():

            # predict logits
            logits_q = net(x_qry, net.parameters(), bn_training=False)

            # calculate the scores based on softmax
            pred_q = F.softmax(logits_q, dim=1)
            start.append(pred_q[:, 1].cpu().numpy())

        # the loss and accuracy after the first update
        if self.update_step_test == 1:

            # predict logits
            logits_q = net(x_qry, fast_weights, bn_training=False)

            # calculate the scores based on softmax
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        else:
            with torch.no_grad():

                # predict logits
                logits_q = net(x_qry, fast_weights, bn_training=False)

                # calculate the scores based on softmax
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)

            for k in range(1, self.update_step_test):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = net(x_spt, fast_weights, bn_training=True)
                if balance_loss:
                    loss = F.cross_entropy(logits, y_spt, weight=torch.tensor([2, 1], device=y_spt.device, dtype=torch.float))
                else:
                    loss = F.cross_entropy(logits, y_spt)

                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)

                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))


                # 执行forward
                logits_q = net(x_qry, fast_weights, bn_training=False)


                with torch.no_grad():
                    # calculate the scores based on softmax
                    pred_q = F.softmax(logits_q, dim=1)
                    print(f"pred_q shape: {pred_q.shape}")
                    print(f"pred_q[:, 1] shape: {pred_q[:, 1].shape}")

            end.append(pred_q[:, 1].cpu().numpy())
        del net

        if return_params:
            return end, fast_weights
        return end

    def get_kshot_data(self, x_spts, y_spts):
        k_shot = 2
        random_index = random.sample(range(k_shot, x_spts.shape[0]), k_shot)
        inputs_x_spts = []
        inputs_y_spts = []
        for index in range(k_shot):
            inputs_x_spts.append(x_spts[index])
            inputs_y_spts.append(y_spts[index])
        for index in random_index:
            inputs_x_spts.append(x_spts[index])
            inputs_y_spts.append(y_spts[index])
        x_spt = torch.stack(inputs_x_spts, dim=0)
        y_spt = torch.stack(inputs_y_spts, dim=0)

        return x_spt, y_spt


    def finetunning_for_ranking(self, peptide, x_spts, y_spts, x_qry, balance_loss=False, epochs=0):
        """
        this is the function used for fine-tuning on support set and test on the query set

        Parameters:
            param peptide: the embedding of peptide
            param x_spt: the embedding of support set
            param y_spt: the labels of support set
            param x_qry: the embedding of query set

        Return:
            the binding scores for the TCRs in the query set in the few-shot setting
        """
        x_spt, y_spt = self.get_kshot_data(x_spts,y_spts)

        querysz = x_qry.size(0)
        start = []
        end = []

        # in order not to ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        if balance_loss:
            loss = F.cross_entropy(logits, y_spt, weight=torch.tensor([2, 1], device=y_spt.device, dtype=torch.float))
        else:
            loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters(), retain_graph=True)
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # the loss and accuracy before first update
        with torch.no_grad():

            # predict logits
            logits_q = net(x_qry, net.parameters(), bn_training=False)

            # calculate the scores based on softmax
            pred_q = F.softmax(logits_q, dim=1)
            start.append(pred_q[:, 1].cpu().numpy())

        # the loss and accuracy after the first update
        if self.update_step_test == 1:

            # predict logits
            logits_q = net(x_qry, fast_weights, bn_training=False)

            # calculate the scores based on softmax
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        else:
            with torch.no_grad():

                # predict logits
                logits_q = net(x_qry, fast_weights, bn_training=False)

                # calculate the scores based on softmax
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)

            for epoch in range(epochs):

                x_spt, y_spt = self.get_kshot_data(x_spts, y_spts)

                for k in range(1, self.update_step_test):
                    # 1. run the i-th task and compute loss for k=1~K-1
                    logits = net(x_spt, fast_weights, bn_training=True)
                    if balance_loss:
                        loss = F.cross_entropy(logits, y_spt, weight=torch.tensor([2, 1], device=y_spt.device, dtype=torch.float))
                    else:
                        loss = F.cross_entropy(logits, y_spt)

                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(loss, fast_weights)

                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                    # predict logits
                    logits_q = net(x_qry, fast_weights, bn_training=False)

                    with torch.no_grad():
                        # calculate the scores based on softmax
                        pred_q = F.softmax(logits_q, dim=1)

            end.append(pred_q[:, 1].cpu().numpy())

        del net

        return end

    def meta_forward_score(self, peptide, x_spt):
        """
        This function is used to perform the zero-shot predition in the condition where you have peptide, TCRs

        Parameters:
            param peptide: the embedding of peptides
            param x_spt: the embedding of TCRs

        Returns:
            the predicted binding scores of the these TCRs
        """
        with torch.no_grad():
            scores = []
            # copy the origin model parameters for the baseline parameter shape
            memory_parameters = deepcopy(self.net.parameters())
            # predict the binding score based on the basis models in the memory block
            for i in range(len(peptide)):
                # retrieve the memory
                r = self.Memory_module.readhead(peptide[i])[0]
                logits = []
                for m, n in enumerate(self.Memory_module.memory.content_memory):
                    # obtain the basis model
                    weights_memory = _split_parameters(n.unsqueeze(0), memory_parameters)
                    logits.append(self.net(x_spt[i], weights_memory, bn_training=False))
                # weighted the predicted result
                pred = sum([r[k] * F.softmax(j) for k, j in enumerate(logits)])
                scores.append(pred[:, 1])
            return scores

    def calculate_query_results(self, query, label, fast_weights):
        # 最后一次需返回logits_q，所以每次都返回了，但只在最后一次使用了
        logits_q = self.net(query, fast_weights, bn_training=True)
        loss_q = F.cross_entropy(logits_q, label)
        with torch.no_grad():
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, label).sum().item()  # convert to numpy  # 正确的个数
        return loss_q, correct, logits_q

    def zero_model_test_few_data(self, peptide, x_spt, y_spt, x_qry):
        end = []
        memory_parameters = deepcopy(self.net.parameters())
        for i in range(len(peptide)):
            r = self.Memory_module.readhead(peptide[i])[0]
            fast_weights = self.Memory_module.memory.content_memory
            # fine-tune
            for k in range(self.update_step_test):
                logits = []
                for m, n in enumerate(fast_weights):
                    weights_memory = _split_parameters(n.unsqueeze(0), memory_parameters)
                    logits.append(self.net(x_spt[i], weights_memory, bn_training=False))
                logits_a = sum([r[k] * j for k, j in enumerate(logits)])  ###
                loss = F.cross_entropy(logits_a, y_spt[i])  ###
                grad = torch.autograd.grad(loss, fast_weights, create_graph=True)  ###
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))[0]  ###

                # fast_weightsl = []
                # for m, n in enumerate(fast_weights):
                #     weights_memory = _split_parameters(n.unsqueeze(0), self.net.parameters())
                #     lossl = F.cross_entropy(logits[m], y_spt[i])
                #     gradl = torch.autograd.grad(lossl, weights_memory, create_graph=True)  ###
                #     fast_weightsl.append(list(map(lambda p: p[1] - self.update_lr * p[0], zip(gradl, weights_memory))))

            # predict
            with torch.no_grad():
                logits_q = []
                for m, n in enumerate(fast_weights):
                    weights_memory = _split_parameters(n.unsqueeze(0), memory_parameters)
                    logits_q.append(self.net(x_qry[i], weights_memory, bn_training=False))
                pred = sum([r[k] * F.softmax(j) for k, j in enumerate(logits_q)])
                end.append(pred[:, 1])
        return end

    def inference_with_params(self, x_qry, finetuned_params):
        """
        使用指定参数进行推理
        Args:
            x_qry: 查询集数据
            finetuned_params: 微调后的模型参数
        """
        net = deepcopy(self.net)
        
        with torch.no_grad():
            logits_q = net(x_qry, finetuned_params, bn_training=False)
            pred_q = F.softmax(logits_q, dim=1)
            end = [pred_q[:, 1].cpu().numpy()]
        
        del net
        return end
    
    def check_frozen_layers(self, net):
        print("\n=== Layer Freezing Status ===")
        for name, param in net.named_parameters():
            print(f"Layer: {name}")
            print(f"Requires grad: {param.requires_grad}")
            print(f"Shape: {param.shape}")
            print("-" * 50)

    def Layer2_Freezing(self, peptide, x_spt, y_spt, x_qry, balance_loss=False, return_params=False):
        """
        Fine-tuning with the first two layers frozen.
        The self-attention layer and first linear layer remain fixed while training other layers.

        Parameters:
            param peptide: the embedding of peptide
            param x_spt: the embedding of support set
            param y_spt: the labels of support set
            param x_qry: the embedding of query set
            param balance_loss: whether to use balanced loss
            param return_params: whether to return the final parameters

        Return:
            the binding scores for the TCRs in the query set in the few-shot setting
        """
        querysz = x_qry.size(0)
        start = []
        end = []

        # Create a copy of the model
        net = deepcopy(self.net)
        
        # Freeze the first two layers
        frozen_count = 0
        for name, param in net.named_parameters():
            if 'self_attention' in name or ('linear' in name and frozen_count < 2):
                param.requires_grad = False
                if 'linear' in name:
                    frozen_count += 1
        self.check_frozen_layers(net)
        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        if balance_loss:
            loss = F.cross_entropy(logits, y_spt, weight=torch.tensor([2, 1], device=y_spt.device, dtype=torch.float))
        else:
            loss = F.cross_entropy(logits, y_spt)
        
        # Only compute gradients for non-frozen parameters
        grad = torch.autograd.grad(loss, filter(lambda p: p.requires_grad, net.parameters()), retain_graph=True)
        
        # Update parameters (keeping frozen layers unchanged)
        fast_weights = []
        grad_idx = 0
        for param in net.parameters():
            if param.requires_grad:
                fast_weights.append(param - self.update_lr * grad[grad_idx])
                grad_idx += 1
            else:
                fast_weights.append(param)

        # The loss and accuracy before first update
        with torch.no_grad():
            logits_q = net(x_qry, net.parameters(), bn_training=False)
            pred_q = F.softmax(logits_q, dim=1)
            start.append(pred_q[:, 1].cpu().numpy())

        # The loss and accuracy after the first update
        if self.update_step_test == 1:
            logits_q = net(x_qry, fast_weights, bn_training=False)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        else:
            with torch.no_grad():
                logits_q = net(x_qry, fast_weights, bn_training=False)
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)

            for k in range(1, self.update_step_test):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = net(x_spt, fast_weights, bn_training=True)
                if balance_loss:
                    loss = F.cross_entropy(logits, y_spt, weight=torch.tensor([2, 1], device=y_spt.device, dtype=torch.float))
                else:
                    loss = F.cross_entropy(logits, y_spt)

                # 2. compute grad on theta_pi (only for non-frozen parameters)
                grad = torch.autograd.grad(loss, filter(lambda p: p.requires_grad, fast_weights))

                # 3. Update parameters (keeping frozen layers unchanged)
                new_fast_weights = []
                grad_idx = 0
                for param in fast_weights:
                    if param.requires_grad:
                        new_fast_weights.append(param - self.update_lr * grad[grad_idx])
                        grad_idx += 1
                    else:
                        new_fast_weights.append(param)
                
                fast_weights = new_fast_weights

                logits_q = net(x_qry, fast_weights, bn_training=False)
                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1)

                end.append(pred_q[:, 1].cpu().numpy())

        del net

        if return_params:
            return end, fast_weights
        return end
    
    def forward_with_adaptation(self, net, x, adaptation_layer, weights=None, bn_training=True):
        """Modified forward pass with adaptation layer"""
        # 使用原始forward获取倒数第二层的输出
        features = net(x, weights[:-2] if weights is not None else None, bn_training=bn_training, return_embedding=True)
        
        # 通过适应层
        x = adaptation_layer(features)
        
        # 通过最后一层
        if weights is not None:
            x = F.linear(x, weights[-2], weights[-1])
        else:
            x = F.linear(x, net.vars[-2], net.vars[-1])
        
        return x
    def LastLayer_Freezing(self, peptide, x_spt, y_spt, x_qry, balance_loss=False, return_params=False):
        """
        Few-shot learning with last layer freezing and adaptation layer
        """
        # 创建保存loss曲线的目录
        if not os.path.exists('./loss_curves'):
            os.makedirs('./loss_curves')
        
        querysz = x_qry.size(0)
        start = []
        end = []

        # 创建模型副本
        net = deepcopy(self.net)

        # 创建和初始化适应层
        # 设置随机种子
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        # 创建和初始化适应层
        self.adaptation_layer = nn.Linear(608, 608)
        torch.nn.init.kaiming_normal_(
            self.adaptation_layer.weight,
            mode='fan_out',
            nonlinearity='relu', 
            a=0.01  # 缩小初始化范围
        )
        torch.nn.init.zeros_(self.adaptation_layer.bias)
        if torch.cuda.is_available():
            self.adaptation_layer = self.adaptation_layer.cuda()
        adaptation_layer = deepcopy(self.adaptation_layer)

        losses = []
        best_loss = float('inf')
        best_weights = None
        best_adaptation_weights = None
        patience = 0
        patience_limit = 10  # 早停阈值
        min_delta = 1e-4    # 最小改善阈值
        
        # 记录第一次更新的loss
        logits = self.forward_with_adaptation(net, x_spt, adaptation_layer)
        if balance_loss:
            loss = F.cross_entropy(logits, y_spt, 
                                weight=torch.tensor([2, 1], device=y_spt.device, dtype=torch.float))
        else:
            loss = F.cross_entropy(logits, y_spt)
                # 计算梯度并更新参数
        grad = torch.autograd.grad(loss, [net.vars[-2], net.vars[-1]] + list(adaptation_layer.parameters()), retain_graph=True)
        
        # 准备fast weights
        fast_weights = list(net.vars[:-2])  # 保持前面层的权重不变
        fast_weights.extend([p - self.update_lr * g for p, g in zip([net.vars[-2], net.vars[-1]], grad[:2])])
        
        # 更新适应层参数
        adaptation_weights = [p - self.update_lr * g for p, g in zip(adaptation_layer.parameters(), grad[2:])]
        adaptation_layer.weight = nn.Parameter(adaptation_weights[0])
        adaptation_layer.bias = nn.Parameter(adaptation_weights[1])
        current_loss = loss.detach().item()  
        losses.append(current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            best_weights = list(fast_weights)
            best_adaptation_weights = [adaptation_layer.weight.data.clone(),
                                     adaptation_layer.bias.data.clone()]

        for k in range(1, self.update_step_test):
            logits = self.forward_with_adaptation(net, x_spt, adaptation_layer, fast_weights)
            if balance_loss:
                loss = F.cross_entropy(logits, y_spt, 
                                    weight=torch.tensor([2, 1], device=y_spt.device, dtype=torch.float))
            else:
                loss = F.cross_entropy(logits, y_spt)
            
            current_loss = loss.detach().item() 
            losses.append(current_loss)
            
            # 更新最佳模型
            if current_loss < (best_loss - min_delta):  # 添加最小改善阈值
                best_loss = current_loss
                best_weights = list(fast_weights)
                best_adaptation_weights = [adaptation_layer.weight.data.clone(),
                                         adaptation_layer.bias.data.clone()]
                patience = 0
            else:
                patience += 1
                
            # 真正的早停
            if patience >= patience_limit:
                print(f"Early stopping at step {k} due to no improvement in loss")
                break
                
            # 计算梯度和更新参数
            grad = torch.autograd.grad(loss, [fast_weights[-2], fast_weights[-1]] + list(adaptation_layer.parameters()))
            new_fast_weights = list(fast_weights[:-2])
            new_fast_weights.extend([p - self.update_lr * g for p, g in zip([fast_weights[-2], fast_weights[-1]], grad[:2])])
            fast_weights = new_fast_weights
            
            adaptation_weights = [p - self.update_lr * g for p, g in zip(adaptation_layer.parameters(), grad[2:])]
            adaptation_layer.weight = nn.Parameter(adaptation_weights[0])
            adaptation_layer.bias = nn.Parameter(adaptation_weights[1])

        # 使用最佳模型进行最终预测
        adaptation_layer.weight.data = best_adaptation_weights[0]
        adaptation_layer.bias.data = best_adaptation_weights[1]
        logits_q = self.forward_with_adaptation(net, x_qry, adaptation_layer, best_weights)
        pred_q = F.softmax(logits_q, dim=1)
        end.append(pred_q[:, 1].cpu().detach().numpy())

        # 保存loss曲线
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title(f'Training Loss (Best Loss: {best_loss:.4f})')                         
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # 创建时间戳文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'./loss_curves/loss_curve_{timestamp}.png')
        plt.close()

        del net
        del adaptation_layer

        if return_params:
            return end, (best_weights, best_adaptation_weights)
        return end
    def inference_with_adaptation(self, x_qry, finetuned_params):
        """
        使用适应层和微调参数进行推理
        
        Args:
            x_qry: 查询集数据
            finetuned_params: 包含网络参数和适应层参数的元组
            
        Returns:
            list: 包含预测概率的列表
        """
        try:
            # 创建模型和适应层的深度复制
            net = deepcopy(self.net)
            adaptation_layer = deepcopy(self.adaptation_layer)
            
            # 参数验证
            if not isinstance(finetuned_params, tuple) or len(finetuned_params) != 2:
                raise ValueError("finetuned_params must be a tuple containing (network_params, adaptation_params)")
            
            # 解包参数
            fast_weights_net, adaptation_params = finetuned_params
            
            # 获取设备信息并移动模型和参数到正确的设备
            device = x_qry.device
            adaptation_layer = adaptation_layer.to(device)
            
            # 更新适应层参数
            adaptation_layer.weight.data = adaptation_params[0].to(device)
            adaptation_layer.bias.data = adaptation_params[1].to(device)
            
            # 设置为评估模式
            net.eval()
            adaptation_layer.eval()
            
            # 执行推理
            with torch.no_grad():
                # 使用修改后的forward_with_adaptation进行预测
                logits = self.forward_with_adaptation(
                    net=net,
                    x=x_qry,
                    adaptation_layer=adaptation_layer,
                    weights=fast_weights_net,
                    bn_training=False
                )
                
                # 计算softmax概率
                pred = F.softmax(logits, dim=1)
                
                # 检查预测结果的有效性
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    print("Warning: Found NaN or Inf values in predictions")
                    return [np.zeros(x_qry.size(0))]
                
                # 返回正类的概率
                return [pred[:, 1].cpu().detach().numpy()]
                
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            traceback.print_exc()  # 打印完整的错误堆栈
            return [np.zeros(x_qry.size(0))]
        finally:
            # 清理内存
            del net
            del adaptation_layer


    def get_kshot_data1(self,x_spts, y_spts, k_shot, offset):
        """
        选择样本，每次调用后更新偏移量
        
        Parameters:
            x_spts: 支持集特征
            y_spts: 支持集标签
            k_shot: 每类样本数量
            offset: 当前偏移量
        
        Returns:
            x_spt: 选择后的特征
            y_spt: 选择后的标签
            new_offset: 更新后的偏移量
        """
        total_samples = x_spts.shape[0]
        
        # 获取前k_shot个样本
        inputs_x_spts = x_spts[0:k_shot]
        inputs_y_spts = y_spts[0:k_shot]
        
        # 获取后k_shot个样本
        inputs_x_spts = torch.cat((inputs_x_spts, x_spts[offset:offset + k_shot]), dim=0)
        inputs_y_spts = torch.cat((inputs_y_spts, y_spts[offset:offset + k_shot]), dim=0)
        
        # 更新偏移量
        new_offset = offset + k_shot
        
        return inputs_x_spts, inputs_y_spts, new_offset
    
    def more_data_finetunning(self, peptide, x_spt, y_spt, x_qry, balance_loss=False, return_params=False,k_shot = 2):
        """
        this is the function used for fine-tuning on support set and test on the query set

        Parameters:
            param peptide: the embedding of peptide
            param x_spt: the embedding of support set
            param y_spt: the labels of support set
            param x_qry: the embedding of query set

        Return:
            the binding scores for the TCRs in the query set in the few-shot setting
        """
        querysz = x_qry.size(0)
        start = []
        end = []
        offset = k_shot
        print(f"offset: {offset}")
        # in order not to ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        x_spt, y_spt, offset = self.get_kshot_data1(x_spt, y_spt, k_shot, offset)
        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        if balance_loss:
            loss = F.cross_entropy(logits, y_spt, weight=torch.tensor([2, 1], device=y_spt.device, dtype=torch.float))
        else:
            loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters(), retain_graph=True)
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # the loss and accuracy before first update
        with torch.no_grad():

            # predict logits
            logits_q = net(x_qry, net.parameters(), bn_training=False)

            # calculate the scores based on softmax
            pred_q = F.softmax(logits_q, dim=1)
            start.append(pred_q[:, 1].cpu().numpy())

        # the loss and accuracy after the first update
        if self.update_step_test == 1:
            
            # predict logits
            logits_q = net(x_qry, fast_weights, bn_training=False)

            # calculate the scores based on softmax
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        else:
            with torch.no_grad():

                # predict logits
                logits_q = net(x_qry, fast_weights, bn_training=False)

                # calculate the scores based on softmax
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)

            for k in range(1, self.update_step_test):
                # 1. run the i-th task and compute loss for k=1~K-1
                x_spt, y_spt, offset = self.get_kshot_data1(x_spt, y_spt, k_shot, offset)
                logits = net(x_spt, fast_weights, bn_training=True)
                if balance_loss:
                    loss = F.cross_entropy(logits, y_spt, weight=torch.tensor([2, 1], device=y_spt.device, dtype=torch.float))
                else:
                    loss = F.cross_entropy(logits, y_spt)

                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)

                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                # predict logits
                logits_q = net(x_qry, fast_weights, bn_training=False)

                with torch.no_grad():
                    # calculate the scores based on softmax
                    pred_q = F.softmax(logits_q, dim=1)

            end.append(pred_q[:, 1].cpu().numpy())

        del net

        if return_params:
            return end, fast_weights
        return end

