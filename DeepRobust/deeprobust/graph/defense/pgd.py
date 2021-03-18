from torch.optim.sgd import SGD
from torch.optim.optimizer import required
from torch.optim import Optimizer
import torch
import sklearn
import numpy as np
import scipy.sparse as sp

class PGD(Optimizer):

    def __init__(self, params, proxs, lr=required, momentum=0, dampening=0, weight_decay=0, alphas=[]):
        defaults = dict(lr=lr, momentum=0, dampening=0,
                        weight_decay=0, nesterov=False)


        super(PGD, self).__init__(params, defaults)

        for group in self.param_groups:
            group.setdefault('proxs', proxs)
            group.setdefault('alphas', alphas)

    def __setstate__(self, state):
        super(PGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('proxs', proxs)
            group.setdefault('alphas', alphas)

    def step(self, delta=0, closure=None):
         for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            proxs = group['proxs']
            alphas = group['alphas']

            # apply the proximal operator to each parameter in a group
            for param in group['params']:
                for prox_operator, alpha in zip(proxs, alphas):
                    # param.data.add_(lr, -param.grad.data)
                    # param.data.add_(delta)
                    param.data = prox_operator(param.data, alpha=alpha*lr)


class ProxOperators():

    def __init__(self):
        self.nuclear_norm = None

    def prox_l1(self, data, alpha):
        data = torch.mul(torch.sign(data), torch.clamp(torch.abs(data)-alpha, min=0))
        return data

    def prox_nuclear(self, data, alpha):
        U, S, V = np.linalg.svd(data.cpu())
        U, S, V = torch.FloatTensor(U), torch.FloatTensor(S), torch.FloatTensor(V)
        self.nuclear_norm = S.sum()
        # print("nuclear norm: %.4f" % self.nuclear_norm)

        diag_S = torch.diag(torch.clamp(S-alpha, min=0))
        return torch.matmul(torch.matmul(U, diag_S), V)

    def prox_nuclear_truncated_2(self, data, alpha, k=50):
        import tensorly as tl
        tl.set_backend('pytorch')
        U, S, V = tl.truncated_svd(data.cpu(), n_eigenvecs=k)
        U, S, V = torch.FloatTensor(U), torch.FloatTensor(S), torch.FloatTensor(V)
        self.nuclear_norm = S.sum()
        # print("nuclear norm: %.4f" % self.nuclear_norm)

        S = torch.clamp(S-alpha, min=0)
        indices = torch.tensor(range(0, U.shape[0]),range(0, U.shape[0])).cuda()
        values = S
        diag_S = torch.sparse.FloatTensor(indices, values, torch.Size(U.shape))
        # diag_S = torch.diag(torch.clamp(S-alpha, min=0))
        U = torch.spmm(U, diag_S)
        V = torch.matmul(U, V)
        return V

    def prox_nuclear_truncated(self, data, alpha, k=200):
        indices = torch.nonzero(data).t()
        values = data[indices[0], indices[1]] # modify this based on dimensionality
        data_sparse = sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()))
        U, S, V = sp.linalg.svds(data_sparse, k=k)
        U, S, V = torch.FloatTensor(U), torch.FloatTensor(S), torch.FloatTensor(V)
        self.nuclear_norm = S.sum()
        diag_S = torch.diag(torch.clamp(S-alpha, min=0))
        return torch.matmul(torch.matmul(U, diag_S), V)

    def prox_nuclear_cuda(self, data, alpha):

        U, S, V = torch.svd(data)
        # self.nuclear_norm = S.sum()
        # print(f"rank = {len(S.nonzero())}")
        S = torch.clamp(S-alpha, min=0)
        indices = torch.tensor([range(0, U.shape[0]),range(0, U.shape[0])])
        values = S
        diag_S = torch.sparse.FloatTensor(indices, values, torch.Size(U.shape))
        # diag_S = torch.diag(torch.clamp(S-alpha, min=0))
        # print(f"rank_after = {len(diag_S.nonzero())}")
        V = torch.spmm(diag_S, V.t_())
        V = torch.matmul(U, V)
        return V


class SGD(Optimizer):


    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

prox_operators = ProxOperators()
