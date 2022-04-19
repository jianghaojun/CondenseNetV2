import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Condensed Convolutional Networks')
    parser.add_argument('--model_name', default='CondenseNetV2-C', help='path to dataset')
    parser.add_argument('--checkpoint_path', default='./', help='path to checkpoint')
    parser.add_argument('--out_path', default='./', help='path to save the output')
    args = parser.parse_args()

    arch = {
        'CondenseNetV2-C': ('4-6-8-10-8', '8-16-32-64-128')
    }
    stages, growth = arch[args.model_name]
    stages = list(map(int, stages.split('-')))
    growth = list(map(int, growth.split('-')))
    total_layers = sum(stages)
    connect_matrix = torch.zeros(total_layers, total_layers)

    model = torch.load(args.checkpoint_path, map_location='cpu')
    state_dict = model['state_dict']

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module'):
            name = k[7:]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v

    layer_count = 1
    each_layer_output_channels = [0, 2 * growth[0]]
    for block_id in range(len(stages)):
        for layer_id in range(stages[block_id]):
            mask_key = 'features.denseblock_{}.denselayer_{}.conv_1._mask'.format(block_id + 1, layer_id + 1)
            weight_key = 'features.denseblock_{}.denselayer_{}.conv_1.conv.weight'.format(block_id + 1, layer_id + 1)
            weight = new_state_dict[weight_key] * new_state_dict[mask_key]
            weight = weight.abs().squeeze()
            weight = weight.sum(0)
            for prev_layer_id in range(layer_count):
                start_channel_id = each_layer_output_channels[prev_layer_id]
                end_channel_id = each_layer_output_channels[prev_layer_id + 1]
                connect_matrix[prev_layer_id, layer_count - 1] = (torch.sum(weight[start_channel_id:end_channel_id]) /
                                                                  (end_channel_id - start_channel_id))

            layer_count += 1
            each_layer_output_channels.append(each_layer_output_channels[-1] + growth[block_id])

    for layer_id in range(layer_count - 1):
        max_v = torch.max(connect_matrix[:, layer_id])
        min_v = torch.min(connect_matrix[:, layer_id])
        connect_matrix[:, layer_id] -= min_v
        connect_matrix[:, layer_id] /= (max_v - min_v)

    connect_matrix = np.array(connect_matrix)
    mask = np.tri(connect_matrix.shape[0], k=-1)
    connect_matrix = np.ma.array(connect_matrix, mask=mask) # mask out the lower triangle

    # plot heatmap
    plt.figure()

    # x, y label
    plt.xlabel('Target Layer')
    plt.ylabel('Source Layer')

    # x ticks
    plt.xticks([0, 1, 5, 10, 15, 20, 25, 30, 35], ['0', '1', '6', '11', '16', '21', '26', '31', '36'])

    # title
    plt.title(args.model_name, loc='center', pad=0)

    # stage separate line
    plt.vlines(x=3.5, ymax=total_layers - 1, ymin=0, colors='grey', linestyles='--')
    plt.vlines(x=9.5, ymax=total_layers - 1, ymin=0, colors='grey', linestyles='--')
    plt.vlines(x=17.5, ymax=total_layers - 1, ymin=0, colors='grey', linestyles='--')
    plt.vlines(x=27.5, ymax=total_layers - 1, ymin=0, colors='grey', linestyles='--')

    # get current axis
    ax = plt.gca()

    cmap = plt.get_cmap('jet', 1000) # jet doesn't have white color
    cmap.set_bad('w') # default value is 'k'
    im = ax.matshow(connect_matrix, cmap=cmap)

    ax.set_aspect(1. / ax.get_data_ratio() - 0.16)

    # ticks position
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    bwith = 2
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)

    # save heatmap figure
    plt.savefig('{}/{}_layer_level_reuse_heatmap.pdf'.format(args.out_path, args.model_name), format='pdf')
    plt.show()



