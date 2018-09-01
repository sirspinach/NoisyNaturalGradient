from collections import OrderedDict

import torch
from torch.autograd import Variable
from torch import nn

from models.base_model import BaseModel
from models.networks import BayesianMultilayer
from optimizers.noisy_adam import NoisyAdam


class FFGModel(BaseModel):
    def __init__(self, opt):
        super(FFGModel, self).__init__(opt)

        self.gpu_ids = opt.gpu_ids
        self.batch_size = opt.batch_size

        self.model = BayesianMultilayer(option=opt.option, gpu_ids=self.gpu_ids, eps=opt.eps, n=opt.n, bias=False)
        if self.gpu_ids:
            self.model.cuda(device=opt.gpu_ids[0])
        self.model_optimizer = NoisyAdam(
            [self.model],
            lr=opt.lr,
        )

        self.result = None

        self.input = self.Tensor(
            opt.batch_size,
            opt.initial_size
        )
        self.label = self.LabelTensor(
            opt.batch_size,
            opt.label_size
        )
        self.loss_function = nn.CrossEntropyLoss()
        self.loss = None

    @property
    def name(self):
        return 'FFGModel'

    def forward(self, volatile=False, is_test=False):
        self.result = self.model(Variable(self.input), is_test=is_test)

    def set_input(self, data):
        self.input.resize_(data[0].size()).copy_(data[0])
        self.label.resize_(data[1].size()).copy_(data[1])

    def get_losses(self):
        # TODO : error occurred during printing loss.
        # TODO : To prevent this error, I add same loss value.
        # TODO : fix this later
        return OrderedDict([
            ('loss', self.loss.cpu().item()),
            ('loss2', self.loss.cpu().item())
        ])

    def get_visuals(self, sample_single_image=True):
        raise NotImplemented

    def save(self, epoch):
        # Torch save only saves parameters.
        # move Bayesian mean values to parameters.
        self.model.save_parameters()
        self.save_network(self.model, self.name + '_model', epoch, self.gpu_ids)

    def remove(self, epoch):
        if epoch == 0:
            return
        self.remove_checkpoint(self.name + '_model', epoch)

    def load(self, epoch):
        self.load_network(self.model, self.name + '_model', epoch)

    def backward(self):
        if self.gpu_ids:
            self.loss = self.loss_function(self.result, Variable(self.label, requires_grad=False).cuda())
        else:
            self.loss = self.loss_function(self.result, Variable(self.label, requires_grad=False))
        try:
            self.loss.backward()
        except:
            print(self.result)
            raise

    def optimize_parameters(self):
        self.forward()

        self.model_optimizer.zero_grad()
        self.backward()
        self.model_optimizer.step()

        # update bayesian params here.
        u_delta_dict, f_dict = self.model_optimizer.get_delta_dicts()
        self.update_bayesian_parameters(u_delta_dict, f_dict)

    def update_bayesian_parameters(self, u_delta_dict, f_dict):
        self.model.update_bayesian_parameters(u_delta_dict, f_dict)

    def test(self):
        self.forward(volatile=True, is_test=True)
        return torch.max(self.result, 1)
