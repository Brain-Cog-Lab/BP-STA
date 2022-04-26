import math

from encoders import Encoder
from operations import *
from common import *
from nodes import *


class ConvNet(nn.Module):
    def __init__(self,
                 step,
                 dataset,
                 num_classes,
                 encode_type,
                 node,
                 *args,
                 **kwargs):
        super(ConvNet, self).__init__()
        self.step = step
        self.dataset = dataset
        self.num_classes = num_classes
        self.node = eval(node) if type(node) == str else node
        self.warm_up = False

        if 'threshold' in kwargs.keys():
            self.threshold = kwargs['threshold']
        else:
            self.threshold = .5
        if 'decay' in kwargs.keys():
            self.decay = kwargs['decay']
        else:
            self.decay = 1.
        self.node = eval(node) if type(node) == str else node

        self.encoder = Encoder(self.step, encode_type)
        if dataset == 'mnist' or dataset == 'fashion':
            self.fun = nn.ModuleList([
                nn.Conv2d(1, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.MaxPool2d(2),

                nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.AvgPool2d(2),

                nn.Flatten(),
                nn.Linear(7 * 7 * 256, 4096),
                nn.ReLU(),
                NDropout(.5),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.Linear(4096, 10 * num_classes),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                VotingLayer(10)
            ])

        elif dataset == 'dvsg' or dataset == 'dvsc10':
            self.fun = nn.ModuleList([
                nn.Conv2d(2, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.MaxPool2d(2),

                nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.MaxPool2d(2),

                nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.MaxPool2d(2),

                nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.AvgPool2d(2),

                nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.AvgPool2d(4),

                nn.Flatten(),
                NDropout(.5),
                nn.Linear(1024, 10 * num_classes),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                VotingLayer(10),
            ])

        elif dataset == 'cifar10' or dataset == 'cifar100':
            self.fun = nn.ModuleList([
                nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.MaxPool2d(2),

                nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.MaxPool2d(2),

                nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.MaxPool2d(2),

                nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.AvgPool2d(4),

                nn.Flatten(),
                nn.Linear(512, 10 * num_classes),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                VotingLayer(10),
            ])

        elif dataset == 'nmnist':
            self.fun = nn.ModuleList([
                nn.Conv2d(2, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.MaxPool2d(2),

                nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.MaxPool2d(2),

                nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.AvgPool2d(2),

                nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                nn.AvgPool2d(4),

                nn.Flatten(),
                NDropout(.5),
                nn.Linear(512, 10 * num_classes),
                self.node(threshold=self.threshold, decay=self.decay) if self.step != 1 else nn.Identity(),
                VotingLayer(10),
            ])

        self.fire_rate = []

    def forward(self, inputs):
        step = self.step if self.warm_up is False or len(inputs.shape) != 4 else 1

        if len(inputs.shape) == 4:
            inputs = self.encoder(inputs)
        else:
            inputs = inputs.permute(1, 0, 2, 3, 4)
            # print(inputs.float())
        self.reset()

        if not self.training:
            self.fire_rate.clear()

        outputs = []
        for t in range(step):
            x = inputs[t]
            if self.dataset == 'dvsg' or self.dataset == 'dvsc10':
                x = F.interpolate(x, size=[64, 64])
            for layer in self.fun:
                if type(layer) == self.node and self.warm_up:
                    continue

                x = layer(x)

                # if hasattr(layer, 'integral'):
                #     print(((x > 0.).float()).sum() / np.product(x.shape))
            outputs.append(x)

            if not self.training:
                self.fire_rate.append(self._get_fire_rate())

        return sum(outputs) / len(outputs)

    def reset(self):
        for layer in self.fun:
            if hasattr(layer, 'n_reset'):
                layer.n_reset()

    def set_ltd(self, value):
        for layer in self.fun:
            if hasattr(layer, 'ltd'):
                layer.ltd = value

    def get_mem_loss(self):
        raise NotImplementedError

    def _get_fire_rate(self):
        outputs = []
        for layer in self.fun:
            if hasattr(layer, 'get_fire_rate'):
                outputs.append(layer.get_fire_rate())
        return outputs

    def get_fire_rate(self):
        x = np.array(self.fire_rate)
        x = x.mean(axis=0)
        return x.tolist()

    def get_threshold(self):
        outputs = []
        for layer in self.fun:
            if hasattr(layer, 'threshold'):
                thresh = nn.Sigmoid()(layer.threshold.detach().clone())
                outputs.append(thresh)
        return outputs

    def get_decay(self):
        outputs = []
        for layer in self.fun:
            if hasattr(layer, 'decay'):
                # outputs.append(float(torch.nn.Sigmoid()(layer.decay.detach().clone())))
                outputs.append(float(layer.decay.detach().clone()))
        return outputs

    def set_warm_up(self, flag):
        self.warm_up = flag
        for mod in self.modules():
            if hasattr(mod, 'set_n_warm_up'):
                mod.set_n_warm_up(flag)
