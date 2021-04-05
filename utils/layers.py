import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, inplanes, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inplanes, inplanes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inplanes // reduction, inplanes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class HS(nn.Module):

    def __init__(self):
        super(HS, self).__init__()
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, inputs):
        return inputs * self.relu6(inputs + 3) / 6


class LGC(nn.Module):
    global_progress = 0.0

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, condense_factor=None,
                 dropout_rate=0., activation='ReLU', bn_momentum=0.1):
        super(LGC, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels, momentum=bn_momentum)
        self.activation_type = activation
        if activation == 'ReLU':
            self.add_module('activation', nn.ReLU(inplace=True))
        elif activation == 'HS':
            self.add_module('activation', HS())
        else:
            raise NotImplementedError
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.drop = nn.Dropout(dropout_rate, inplace=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups=1, bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.condense_factor = condense_factor
        if self.condense_factor is None:
            self.condense_factor = self.groups
        ### Parameters that should be carefully used
        self.register_buffer('_count', torch.zeros(1))
        self.register_buffer('_stage', torch.zeros(1))
        self.register_buffer('_mask', torch.ones(self.conv.weight.size()))
        ### Check if arguments are valid
        assert self.in_channels % self.groups == 0, "group number can not be divided by input channels"
        assert self.in_channels % self.condense_factor == 0, "condensation factor can not be divided by input channels"
        assert self.out_channels % self.groups == 0, "group number can not be divided by output channels"

    def forward(self, x):
        self._check_drop()
        x = self.norm(x)
        x = self.activation(x)
        if self.dropout_rate > 0:
            x = self.drop(x)
        ### Masked output
        weight = self.conv.weight * self.mask
        return F.conv2d(x, weight, None, self.conv.stride,
                        self.conv.padding, self.conv.dilation, 1)

    def _check_drop(self):
        progress = LGC.global_progress
        delta = 0
        if progress * 2 < (1 + 1e-3):
            ### Get current stage
            for i in range(self.condense_factor - 1):
                if progress * 2 < (i + 1) / (self.condense_factor - 1):
                    stage = i
                    break
            else:
                stage = self.condense_factor - 1
            ### Check for dropping
            if not self._reach_stage(stage):
                self.stage = stage
                delta = self.in_channels // self.condense_factor
            if delta > 0:
                self._dropping(delta)
        return

    def _dropping(self, delta):
        print('LearnedGroupConv dropping')
        weight = self.conv.weight * self.mask
        ### Sum up all kernels
        ### Assume only apply to 1x1 conv to speed up
        assert weight.size()[-1] == 1
        weight = weight.abs().squeeze()
        assert weight.size()[0] == self.out_channels
        assert weight.size()[1] == self.in_channels
        d_out = self.out_channels // self.groups
        ### Shuffle weight
        weight = weight.view(d_out, self.groups, self.in_channels)
        weight = weight.transpose(0, 1).contiguous()
        weight = weight.view(self.out_channels, self.in_channels)
        ### Sort and drop
        for i in range(self.groups):
            wi = weight[i * d_out:(i + 1) * d_out, :]
            ### Take corresponding delta index
            di = wi.sum(0).sort()[1][self.count:self.count + delta]
            for d in di.data:
                self._mask[i::self.groups, d, :, :].fill_(0)
        self.count = self.count + delta

    @property
    def count(self):
        return int(self._count[0])

    @count.setter
    def count(self, val):
        self._count.fill_(val)

    @property
    def stage(self):
        return int(self._stage[0])

    @stage.setter
    def stage(self, val):
        self._stage.fill_(val)

    @property
    def mask(self):
        return self._mask

    def _reach_stage(self, stage):
        return (self._stage >= stage).all()

    @property
    def lasso_loss(self):
        if self._reach_stage(self.groups - 1):
            return 0
        weight = self.conv.weight * self.mask
        ### Assume only apply to 1x1 conv to speed up
        assert weight.size()[-1] == 1
        weight = weight.squeeze().pow(2)
        d_out = self.out_channels // self.groups
        ### Shuffle weight
        weight = weight.view(d_out, self.groups, self.in_channels)
        weight = weight.sum(0).clamp(min=1e-6).sqrt()
        return weight.sum()


class SFR(nn.Module):
    global_progress = 0.0

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, condense_factor=None,
                 dropout_rate=0., activation='ReLU', bn_momentum=0.1):
        super(SFR, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels, momentum=bn_momentum)
        self.activation_type = activation
        if activation == 'ReLU':
            self.add_module('activation', nn.ReLU(inplace=True))
        elif activation == 'HS':
            self.add_module('activation', HS())
        else:
            raise NotImplementedError
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.drop = nn.Dropout(dropout_rate, inplace=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups=1, bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.condense_factor = condense_factor
        if self.condense_factor is None:
            self.condense_factor = self.groups
        ### Parameters that should be carefully used
        self.register_buffer('_count', torch.zeros(1))
        self.register_buffer('_stage', torch.zeros(1))
        self.register_buffer('_mask', torch.ones(self.conv.weight.size()))
        ### Check if arguments are valid
        assert self.in_channels % self.groups == 0, "group number can not be divided by input channels"
        assert self.out_channels % self.condense_factor == 0, "transpose factor can not be divided by input channels"
        assert self.out_channels % self.groups == 0, "group number can not be divided by output channels"

        self._init_weight()

    def forward(self, x):
        self._check_drop()
        x = self.norm(x)
        x = self.activation(x)
        if self.dropout_rate > 0:
            x = self.drop(x)
        ### Masked output
        weight = self.conv.weight * self.mask
        return F.conv2d(x, weight, None, self.conv.stride,
                        self.conv.padding, self.conv.dilation, 1)

    def _check_drop(self):
        progress = SFR.global_progress
        delta = 0
        if progress * 2 < (1 + 1e-3):
            ### Get current stage
            for i in range(self.condense_factor - 1):
                if progress * 2 < (i + 1) / (self.condense_factor - 1):
                    stage = i
                    break
            else:
                stage = self.condense_factor - 1
            ### Check for dropping
            if not self._reach_stage(stage):
                self.stage = stage
                delta = self.out_channels // self.condense_factor
            if delta > 0:
                self._dropping(delta)
        return

    def _dropping(self, delta):
        print('LearnedGroupConvTrans dropping')
        weight = self.conv.weight * self.mask
        ### Sum up all kernels
        ### Assume only apply to 1x1 conv to speed up
        assert weight.size()[-1] == 1
        weight = weight.abs().squeeze()
        assert weight.size()[0] == self.out_channels
        assert weight.size()[1] == self.in_channels
        d_in = self.in_channels // self.groups
        ### Shuffle weight
        weight = weight.view(self.out_channels, d_in, self.groups)
        weight = weight.transpose(1, 2).contiguous()
        weight = weight.view(self.out_channels, self.in_channels)
        ### Sort and drop
        for i in range(self.groups):
            wi = weight[:, i * d_in:(i + 1) * d_in]
            ### Take corresponding delta index
            di = wi.sum(1).sort()[1][self.count:self.count + delta]
            for d in di.data:
                self._mask[d, i::self.groups, :, :].fill_(0)
        self.count = self.count + delta

    @property
    def count(self):
        return int(self._count[0])

    @count.setter
    def count(self, val):
        self._count.fill_(val)

    @property
    def stage(self):
        return int(self._stage[0])

    @stage.setter
    def stage(self, val):
        self._stage.fill_(val)

    @property
    def mask(self):
        return self._mask

    def _reach_stage(self, stage):
        return (self._stage >= stage).all()

    @property
    def lasso_loss(self):
        if self._reach_stage(self.groups - 1):
            return 0
        weight = self.conv.weight * self.mask
        ### Assume only apply to 1x1 conv to speed up
        assert weight.size()[-1] == 1
        weight = weight.squeeze().pow(2)
        d_in = self.in_channels // self.groups
        ### Shuffle weight
        weight = weight.view(self.out_channels, d_in, self.groups)
        weight = weight.sum(1).clamp(min=1e-6).sqrt()
        return weight.sum()

    def _init_weight(self):
        self.norm.weight.data.fill_(0)
        self.norm.bias.data.zero_()


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1, activation='ReLU', bn_momentum=0.1):
        super(Conv, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels, momentum=bn_momentum))
        if activation == 'ReLU':
            self.add_module('activation', nn.ReLU(inplace=True))
        elif activation == 'HS':
            self.add_module('activation', HS())
        else:
            raise NotImplementedError
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding, bias=False,
                                          groups=groups))


def ShuffleLayer(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    ### reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    ### transpose
    x = torch.transpose(x, 1, 2).contiguous()
    ### reshape
    x = x.view(batchsize, -1, height, width)
    return x


def ShuffleLayerTrans(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    ### reshape
    x = x.view(batchsize, channels_per_group, groups, height, width)
    ### transpose
    x = torch.transpose(x, 1, 2).contiguous()
    ### reshape
    x = x.view(batchsize, -1, height, width)
    return x


class CondensingLGC(nn.Module):
    def __init__(self, model):
        super(CondensingLGC, self).__init__()
        layer_str = str(model)
        type_name = layer_str[:layer_str.find('(')].strip()
        self.typename = type_name
        self.in_channels = model.conv.in_channels \
                           * model.groups // model.condense_factor
        self.out_channels = model.conv.out_channels
        self.groups = model.groups
        self.condense_factor = model.condense_factor
        self.norm = nn.BatchNorm2d(self.in_channels)

        # self.relu = nn.ReLU(inplace=True)
        if model.activation_type == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif model.activation_type == 'HS':
            self.activation = HS()
        else:
            raise NotImplementedError

        self.conv = nn.Conv2d(self.in_channels, self.out_channels,
                              kernel_size=model.conv.kernel_size,
                              padding=model.conv.padding,
                              groups=self.groups,
                              bias=False,
                              stride=model.conv.stride)
        self.register_buffer('index', torch.LongTensor(self.in_channels))
        index = 0
        mask = model._mask.mean(-1).mean(-1)
        ## comments:  mask.sum(1) = self.gtoups. the mask is shuffled weight
        for i in range(self.groups):
            for j in range(model.conv.in_channels):
                if index < (self.in_channels // self.groups) * (i + 1) and mask[
                    i, j] == 1:  # pattern is same inside group
                    for k in range(self.out_channels // self.groups):
                        idx_i = int(k + i * (self.out_channels // self.groups))
                        idx_j = index % (self.in_channels // self.groups)
                        self.conv.weight.data[idx_i, idx_j, :, :] = \
                            model.conv.weight.data[int(i + k * self.groups), j, :, :]
                        self.norm.weight.data[index] = model.norm.weight.data[j]
                        self.norm.bias.data[index] = model.norm.bias.data[j]
                        self.norm.running_mean[index] = model.norm.running_mean[j]
                        self.norm.running_var[index] = model.norm.running_var[j]
                    self.index[index] = j
                    index += 1

    def forward(self, x):
        x = torch.index_select(x, 1, self.index)
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        x = ShuffleLayer(x, self.groups)
        return x


class CondensingSFR(nn.Module):
    def __init__(self, model):
        super(CondensingSFR, self).__init__()
        layer_str = str(model)
        type_name = layer_str[:layer_str.find('(')].strip()
        self.typename = type_name
        self.in_channels = model.conv.in_channels
        self.out_channels = model.conv.out_channels \
                            * model.groups // model.condense_factor

        self.groups = model.groups
        self.condense_factor = model.condense_factor
        self.norm = nn.BatchNorm2d(self.in_channels)

        # self.relu = nn.ReLU(inplace=True)
        if model.activation_type == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif model.activation_type == 'HS':
            self.activation = HS()
        else:
            raise NotImplementedError

        self.conv = nn.Conv2d(self.in_channels, self.out_channels,
                              kernel_size=model.conv.kernel_size,
                              padding=model.conv.padding,
                              groups=self.groups,
                              bias=False,
                              stride=model.conv.stride)
        self.register_buffer('index', torch.zeros(self.out_channels, self.out_channels))

        out_index = torch.zeros(self.groups)
        mask = model._mask.mean(-1).mean(-1)

        for i in range(model.conv.out_channels):
            for j in range(self.groups):
                if out_index[j] < (self.out_channels // self.groups) and mask[i, j] == 1:
                    for k in range(self.in_channels // self.groups):
                        idx_i = int(out_index[j] + j * (self.out_channels // self.groups))  # out_channel
                        idx_j = k  # in_channel
                        self.conv.weight.data[idx_i, idx_j, :, :] = \
                            model.conv.weight.data[i, int(j + k * self.groups), :, :]
                        self.index[idx_i, i] = 1.0
                    out_index[j] += 1

        self.norm.weight.data = model.norm.weight.data
        self.norm.bias.data = model.norm.bias.data
        self.norm.running_mean = model.norm.running_mean
        self.norm.running_var = model.norm.running_var

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = ShuffleLayerTrans(x, self.groups)
        x = self.conv(x)  # SIZE: N, C, H, W
        N, C, H, W = x.size()
        x = x.view(N, C, H * W)
        x = x.transpose(1, 2).contiguous()  # SIZE: N, HW, C
        x = torch.matmul(x, self.index)  # x SIZE: N, HW, C; self.index SIZE: C, C; OUTPUT SIZE: N, HW, C
        x = x.transpose(1, 2).contiguous()  # SIZE: N, C, HW
        x = x.view(N, C, H, W)  # SIZE: N, C, HW
        return x


class CondenseLGC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1, activation='ReLU'):
        super(CondenseLGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.norm = nn.BatchNorm2d(self.in_channels)
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'HS':
            self.activation = HS()
        else:
            raise NotImplementedError
        self.conv = nn.Conv2d(self.in_channels, self.out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=self.groups,
                              bias=False)
        self.register_buffer('index', torch.LongTensor(self.in_channels))
        self.index.fill_(0)

    def forward(self, x):
        x = torch.index_select(x, 1, self.index)
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        x = ShuffleLayer(x, self.groups)
        return x


class CondenseSFR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1, activation='ReLU'):
        super(CondenseSFR, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.norm = nn.BatchNorm2d(self.in_channels)
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'HS':
            self.activation = HS()
        else:
            raise NotImplementedError
        self.conv = nn.Conv2d(self.in_channels, self.out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              groups=self.groups,
                              bias=False,
                              stride=stride)
        self.register_buffer('index', torch.zeros(self.out_channels, self.out_channels))

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = ShuffleLayerTrans(x, self.groups)
        x = self.conv(x)  # SIZE: N, C, H, W
        N, C, H, W = x.size()
        x = x.view(N, C, H * W)
        x = x.transpose(1, 2).contiguous()  # SIZE: N, HW, C
        x = torch.matmul(x, self.index)  # x SIZE: N, HW, C; self.index SIZE: C, C; OUTPUT SIZE: N, HW, C
        x = x.transpose(1, 2).contiguous()  # SIZE: N, C, HW
        x = x.view(N, C, H, W)  # SIZE: N, C, HW
        return x
