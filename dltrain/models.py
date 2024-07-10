from copy import deepcopy

import torch
from torch.nn import Flatten
from torchvision import models
import numpy as np
import math

custom_resnet_layers_to_freeze = [
    [],
    ['conv1', 'bn1', 'relu', 'maxpool'],
]
for i in range(1,5):
    custom_resnet_layers_to_freeze.append(
        custom_resnet_layers_to_freeze[-1] + [f'layer{i}']
    )


class CustomResnet(models.ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = Flatten(1)

        self.ordered_keys = ['conv1', 'bn1', 'relu', 'maxpool',
                             'layer1', 'layer2', 'layer3', 'layer4',
                             'avgpool', 'flatten', 'fc'
                             ]
        self.todrop_list = []

    def drop_keys(self, todrop_list):
        assert np.isin(todrop_list, self.ordered_keys).all()
        self.todrop_list = todrop_list

    def add_after(self, previous_layer_name, new_layer, new_layer_name):
        assert previous_layer_name in self.ordered_keys
        assert new_layer_name not in self.ordered_keys
        assert new_layer_name not in self.todrop_list

        self.ordered_keys.insert(
            self.ordered_keys.index(previous_layer_name) + 1,
            new_layer_name)
        self.add_module(new_layer_name, new_layer)

    def forward(self, x):
        for k in self.ordered_keys:
            if k in self.todrop_list:
                continue
            f = getattr(self, k)
            x = f(x)

        return x


class Identity(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class ConditionalResnetEmbedding(CustomResnet):
    def __init__(self, *args, num_classes=None, cmap=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmap = torch.nn.Sequential(
            torch.nn.Linear(num_classes, cmap),
            torch.nn.LeakyReLU()) if cmap is not None else Identity()

    def forward(self, x, c):
        for k in self.ordered_keys[:-1]:
            if k in self.todrop_list:
                continue
            f = getattr(self, k)
            x = f(x)
        if 'fc' not in self.todrop_list:
            x = torch.cat((x, self.cmap(c)), 1)
            x = self.fc(x)
        return x

class ConditionalResnetPointwise(CustomResnet):
    def __init__(self, *args, num_classes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmap = torch.nn.Sequential(
            torch.nn.Linear(num_classes, 256),
            torch.nn.LeakyReLU())

    def forward(self, x, c):
        for k in self.ordered_keys[:-1]:
            if k in self.todrop_list:
                continue
            f = getattr(self, k)
            x = f(x)
        if 'fc' not in self.todrop_list:
            x = x * self.cmap(c)
            x = self.fc(x)
        return x


class MultiHeadResnet(CustomResnet):
    def __init__(self, *args, num_classes=None, out_classes=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        for i in range(num_classes):
            setattr(self, f'fc_{i}', torch.nn.Sequential(
                torch.nn.Linear(self.fc.in_features, out_classes),
                torch.nn.LeakyReLU()))
        del self.fc

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        if 'fc_0.weight' not in state_dict and 'fc.weight' in state_dict and not strict:
            for i in range(self.num_classes):
                state_dict[f'fc_{i}.weight'] = deepcopy(state_dict['fc.weight'])
                state_dict[f'fc_{i}.bias'] = deepcopy(state_dict['fc.bias'])

            del state_dict['fc.weight'], state_dict['fc.bias']

        super().load_state_dict(state_dict, strict)


    def forward(self, x, c):
        for k in self.ordered_keys[:-1]:
            if k in self.todrop_list:
                continue
            f = getattr(self, k)
            x = f(x)

        # OHE to number
        c = c.argmax(1)
        # c_inv For later reversing
        c_inv = torch.zeros_like(c)

        # out stores the output of each classifier
        out = []
        # total stores the total number of instances seen
        total = 0

        for i in torch.arange(self.num_classes,device=c.device):
            msk = c == i
            if msk.any():
                class_total = msk.sum()
                c_inv[msk] = torch.arange(class_total, device=c_inv.device) + total
                total += class_total
                out.append(getattr(self,f'fc_{i}')(x[msk]))

        return torch.cat(out)[c_inv]



class ConditionalResnetConvolutional(CustomResnet):
    def __init__(self, *args, num_classes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(
            in_channels=self.conv1.in_channels + num_classes,
            out_channels=self.conv1.out_channels,
            kernel_size=self.conv1.kernel_size,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
            dilation=self.conv1.dilation,
            groups=self.conv1.groups,
            bias=self.conv1.bias is not None
        )

    def forward(self, x, c):
        assert (x.size(0) == c.size(0))
        x = torch.cat((
            x,
            c[:, :, None, None].expand((c.size(0), c.size(1), *x.shape[2:]))
        ), 1)
        for k in self.ordered_keys:
            if k in self.todrop_list:
                continue
            f = getattr(self, k)
            x = f(x)
        return x


def get_custom_resnet18(fc_out=2, input_side=32, pretrained=True, **kwargs):
    model = models.resnet18(pretrained=pretrained)
    model_sd = model.state_dict()

    custom_model = CustomResnet(models.resnet.BasicBlock, [2, 2, 2, 2])
    custom_model.load_state_dict(model_sd)

    custom_model.drop_keys(['layer4', 'fc'])
    fake_inp = torch.zeros((1, 3, input_side, input_side))
    fc_size = custom_model(fake_inp).size(1)
    # print(fc_size)
    custom_model.drop_keys(['layer4'])
    custom_model.fc = torch.nn.Linear(fc_size, fc_out)

    return custom_model


# Concatenate on embedding
# https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Age_ProgressionRegression_by_CVPR_2017_paper.pdf
def get_conditional_resnet18_embedding(fc_out=2, input_side=32, pretrained=True, num_classes=30, cmap=None, **kwargs):
    model = models.resnet18(pretrained=pretrained)
    model_sd = model.state_dict()

    custom_model = ConditionalResnetEmbedding(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=30, cmap=cmap)
    custom_model.load_state_dict(model_sd, strict=False)

    custom_model.drop_keys(['layer4', 'fc'])
    fake_inp = torch.zeros((1, 3, input_side, input_side))
    fc_size = custom_model(fake_inp, torch.zeros((1, 0))).size(1) + (num_classes if cmap is None else cmap)
    # print(fc_size)
    custom_model.drop_keys(['layer4'])
    custom_model.fc = torch.nn.Linear(fc_size, fc_out)

    return custom_model

def get_multihead_resnet18(fc_out=2, input_side=32, pretrained=True, num_classes=30,  **kwargs):
    model = models.resnet18(pretrained=pretrained)
    model_sd = model.state_dict()

    custom_model = MultiHeadResnet(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    custom_model.load_state_dict(model_sd, strict=False)

    return custom_model

# Pointwise multiplication before FC (like StyleGAN discriminator)
def get_conditional_resnet18_pointwise(fc_out=2, input_side=32, pretrained=True, num_classes=30, dropout=False, **kwargs):
    model = models.resnet18(pretrained=pretrained)
    model_sd = model.state_dict()

    custom_model = ConditionalResnetPointwise(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    custom_model.load_state_dict(model_sd, strict=False)

    if dropout:
        custom_model.add_after('avgpool', torch.nn.Dropout2d(0.5), 'dropout')

    custom_model.drop_keys(['layer4', 'fc'])
    fake_inp = torch.zeros((1, 3, input_side, input_side))
    fc_size = custom_model(fake_inp, torch.zeros((1, 0))).size(1)
    # print(fc_size)
    custom_model.drop_keys(['layer4'])
    custom_model.fc = torch.nn.Linear(fc_size, fc_out)

    return custom_model

# Concatenate on input

def get_conditional_resnet18_convolution(fc_out=2, input_side=32, pretrained=True, num_classes=30, cmap=None, **kwargs):
    model = models.resnet18(pretrained=pretrained)
    model_sd = model.state_dict()

    # Match conv1
    conv1_insize = num_classes + 3
    model_sd['conv1.weight'] = model_sd['conv1.weight'].tile(1, math.ceil(conv1_insize / 3), 1, 1)[:,:conv1_insize]

    custom_model = ConditionalResnetConvolutional(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=30)
    custom_model.load_state_dict(model_sd, strict=True)

    custom_model.drop_keys(['layer4', 'fc'])
    fake_inp = torch.zeros((1, 3, input_side, input_side))
    fc_size = custom_model(fake_inp, torch.zeros((1, num_classes))).size(1)
    # print(fc_size)
    custom_model.drop_keys(['layer4'])
    custom_model.fc = torch.nn.Linear(fc_size, fc_out)

    return custom_model


def get_custom_vgg11_bn(fc_out=2, input_side=32, pretrained=True, **kwargs):
    vgg_model = models.vgg11_bn(pretrained=pretrained)
    vgg_model.classifier = Identity()
    fake_inp = torch.zeros((1, 3, input_side, input_side))
    fc_size = vgg_model(fake_inp).size(1)
    # print(fc_size)
    vgg_model.classifier = torch.nn.Linear(fc_size, fc_out)
    return vgg_model


