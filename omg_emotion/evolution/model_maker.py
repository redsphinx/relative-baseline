import numpy as np
import torch
from torch.nn.functional import conv3d

from relative_baseline.omg_emotion.convttn3d import ConvTTN3d


def make_conv_layer(project_variable, which_layer, k, p, conv_type):
    # start counting at 1
    assert which_layer > 0

    if which_layer == 1:
        the_in_channels = project_variable.num_in_channels
    else:
        the_in_channels = project_variable.num_out_channels[which_layer - 2]

    the_out_channels = project_variable.num_out_channels[which_layer - 1]

    if conv_type == 'conv3dttn':
        return ConvTTN3d(in_channels=the_in_channels,
                         out_channels=the_out_channels,
                         kernel_size=k,
                         padding=p,
                         project_variable=project_variable)
    else:
        return torch.nn.Conv3d(in_channels=the_in_channels,
                               out_channels=the_out_channels,
                               kernel_size=k,
                               padding=p)

def make_pool_layer(is_last, not_last, last):

    if is_last:
        pool_type = last
    else:
        pool_type = not_last

    if pool_type == 'max':
        return torch.nn.MaxPool3d(kernel_size=2)
    else:
        return torch.nn.AvgPool3d(kernel_size=2)


def correct_names(list_of_names):
    types = ['conv', 'pool', 'fc']
    counting_types = [1, 1, 1]
    new_list = []

    for name in list_of_names:
        new_list.append('%s%d' % (name, counting_types[types.index(name)]))
        counting_types[types.index(name)] = counting_types[types.index(name)] + 1

    return new_list


class ModularConv(torch.nn.Module):

    def __init__(self, project_variable):
        super(ModularConv, self).__init__()

        settings = project_variable.genome

        self.conv_layers = {}
        for i in range(settings['num_conv_layers']):
            conv = make_conv_layer(project_variable=project_variable,
                                   which_layer=i+1,
                                   k=settings['kernel_size_per_layer'][i],
                                   p=settings['padding'][i],
                                   conv_type=settings['conv_layer_type'][i])
            self.conv_layers['conv%d' % (i+1)] = conv

        self.pool_layers = {}

        num_pooling_layers = settings['architecture_order'].count('pool')
        for i in range(num_pooling_layers):
            last = False
            if i == num_pooling_layers-1:
                last = True

            pool = make_pool_layer(last, settings['pooling_after_conv'], settings['pooling_final'])
            self.pool_layers['pool%d' % (i+1)] = pool

        in_features = settings['in_features']

        self.fc_layers = {'fc1': torch.nn.Linear(in_features, settings['fc_layer']),
                          'fc2': torch.nn.Linear(settings['fc_layer'], project_variable.label_size)}


    def forward(self, x, device, settings):

        types_architecture = settings['architecture_order']
        names_architecture = correct_names(types_architecture)

        for i, l in enumerate(types_architecture):

            if l == 'conv':
                x = self.conv_layers[names_architecture[i]](x, device)
                x = torch.nn.functional.relu(x)

            elif l == 'pool':
                x = self.pool_layers[names_architecture[i]](x)

            elif l == 'fc':
                x = self.fc_layers[names_architecture[i]](x)
                if [names_architecture[i]] == 'fc1':
                    x = torch.nn.functional.relu(x)

        return x
