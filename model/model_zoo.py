import torch
from torch import nn
import torch.nn.functional as F


def padding_same(input,  kernel, stride=1, dilation=1):
    """
        Calculates padding for applied dilation.
    """
    return int(0.5 * (stride * (input - 1) - input + kernel + (dilation - 1) * (kernel - 1)))


class ClassicConv(nn.Module):
    r"""Convolution network."""

    def __init__(
        self, segment_sz, kernels=None, in_channels=None, out_channels=None,
        conv_groups=None, n_conv_layers_per_block=1, n_blocks=2,
        pooling_off=False, n_classes=None, low_conv_options=None,
        shortcut_conn=False, log=print
    ):
        r"""Instance of convnet."""
        super(ClassicConv, self).__init__()

        log(
            f"segment_sz:{segment_sz}, kernels:{kernels}, in-chan:{in_channels}, "
            f"out-chan:{out_channels}, conv-gr:{conv_groups}, "
            f"n-conv-layer-per-block:{n_conv_layers_per_block}, "
            f"n_block:{n_blocks}, n_class:{n_classes}, "
            f"low-conv:{low_conv_options}, shortcut:{shortcut_conn}")

        self.iter = 0
        self.log = log
        self.input_sz = self.segment_sz = segment_sz
        self.kernels = kernels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_groups = conv_groups
        self.n_conv_layers_per_block = n_conv_layers_per_block
        self.n_blocks = n_blocks
        self.low_conv_options = low_conv_options
        self.shortcut_conn = shortcut_conn
        self.pooling_off = pooling_off
        self.n_classes = n_classes

        self.input_bn = nn.BatchNorm1d(self.in_channels)
        self.low_conv = self.make_low_conv()
        # self.features = self.make_layers_deep()

        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.pool = nn.ModuleList([])
        self.shortcut = nn.ModuleList([])
        self.make_layers_deep()

        # n_hidden = self.calculate_hidden()
        # self.classifier = nn.Sequential(
        #     nn.Linear(n_hidden, n_hidden//2),
        #     nn.BatchNorm1d(n_hidden//2),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(n_hidden//2, 2)
        # )
        # self.gap = nn.AdaptiveAvgPool1d(1)
        self.gap = None
        # self.classifier = nn.Linear(self._out_channels, n_classes)
        # self.classifier = None

    def name(self):
        return (
            f"{self.__class__.__name__}_"
            f"segsz{self.segment_sz}_scut{self.shortcut_conn}_"
            f"lccfg{'x'.join(str(x) for x in self.low_conv_options['cfg'])}_"
            f"lckr{'x'.join(str(x) for x in self.low_conv_options['kernel'])}_"
            f"lcst{'x'.join(str(x) for x in self.low_conv_options['stride'])}_"
            f"lccg{'x'.join(str(x) for x in self.low_conv_options['conv_groups'])}_"
            f"blk{self.n_blocks}_cpblk{self.n_conv_layers_per_block}_"
            # f"kr{'x'.join(str(x) for x in self.kernels)}_"
            f"kr{self.kernels[0]}x{len(self.kernels)}_"
            # f"och{'x'.join(str(x) for x in self.out_channels)}_"
            f"och{self.out_channels[0]}x{len(self.out_channels)}_"
            # f"cg{'x'.join(str(x) for x in self.conv_groups)}"
            f"cg{self.conv_groups[0]}x{len(self.conv_groups)}"
        )
        # return f"{self.__class__.__name__}"

    def forward(self, x):
        self.debug(f'  input: {x.shape}')

        x = self.input_bn(x)

        out = self.low_conv(x)
        self.debug(f'  low_conv out: {out.shape}')

        # out = self.features(x)
        # self.debug(f'features out: {out.shape}')

        for i_blk in range(self.n_blocks):
            if self.shortcut_conn:
                out = self.shortcut[i_blk](out)
                self.debug(f"[block:{i_blk}] shortcut out: {out.shape}")
            else:
                for i_conv_blk in range(self.n_conv_layers_per_block):
                    idx_flat = self.n_conv_layers_per_block\
                        * i_blk+i_conv_blk
                    self.debug(
                        f"  block({i_blk}) conv({i_conv_blk}) {self.conv[idx_flat]}")
                    out = self.conv[idx_flat](out)
                    out = self.bn[idx_flat](out)
                    r"Per layer activation."
                    # out = self.act[idx_flat](out)
                r"End of block activation."
                out = self.act[i_blk](out)
                self.debug(
                    f"  block({i_blk}) out:{out.shape}")
            r"One less pooling layer."
            if not self.pooling_off and i_blk < self.n_blocks - 1:
                out = self.pool[i_blk](out)
                self.debug(f"  block({i_blk}) pool-out:{out.shape}")

        if self.gap:
            out = self.gap(out)
            self.debug(f"  GAP out: {out.shape}")

        # out = out.view(out.size(0), -1)
        # self.debug(f'  flatten: {out.shape}')
        out = self.classifier(out)
        self.debug(f'  out: {out.shape}')
        self.iter += 1
        return out

    def calculate_hidden(self):
        return self.input_sz * self.out_channels[-1]

    def make_layers_deep(self):
        # layers = []
        in_channels = self.low_conv_hidden_dim
        for i in range(self.n_blocks):
            self._out_channels = self.out_channels[i]
            layers_for_shortcut = []
            in_channel_for_shortcut = in_channels
            for _ in range(self.n_conv_layers_per_block):
                self.conv.append(
                    nn.Conv1d(
                        in_channels,
                        self._out_channels,
                        kernel_size=self.kernels[i],
                        groups=self.conv_groups[i],
                        # Disable bias in convolutional layers before batchnorm.
                        bias=False,
                        padding=padding_same(
                            input=self.input_sz,
                            kernel=self.kernels[i],
                            stride=1,
                            dilation=1)
                    ))
                self.bn.append(nn.BatchNorm1d(self._out_channels))
                self.act.append(
                    # nn.ReLU(inplace=True)
                    nn.ELU(inplace=True, alpha=2.0)
                )
                if self.shortcut_conn:
                    layers_for_shortcut.extend([
                        self.conv[-1], self.bn[-1],
                    ])
                in_channels = self._out_channels

            if self.shortcut_conn:
                self.shortcut.append(
                    ShortcutBlock(
                        layers=nn.Sequential(*layers_for_shortcut),
                        in_channels=in_channel_for_shortcut,
                        out_channels=self._out_channels,
                        point_conv_group=self.conv_groups[i])
                )
            if not self.pooling_off and i < self.n_blocks - 1:
                self.pool.append(nn.MaxPool1d(2, stride=2))
                self.input_sz //= 2
        r"If shortcut_conn is true, empty conv, and bn module-list. \
        This may be necessary to not to calculate gradients for the \
        same layer twice."
        if self.shortcut_conn:
            self.conv = nn.ModuleList([])
            self.bn = nn.ModuleList([])
            self.act = nn.ModuleList([])

        r"Scoring layer"
        self.classifier = nn.Conv1d(
            in_channels, self.n_classes, kernel_size=1, padding=0,
            dilation=1
        )

    def make_low_conv(self):
        layers = []
        count_pooling = 0
        i_kernel = 0
        in_chan = self.in_channels
        # input_sz = self.input_sz
        for x in self.low_conv_options["cfg"]:
            if x == 'M':
                count_pooling += 1
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
                self.input_sz /= 2
            else:
                conv = nn.Conv1d(
                    in_channels=in_chan, out_channels=x,
                    kernel_size=self.low_conv_options["kernel"][i_kernel],
                    groups=self.low_conv_options["conv_groups"][i_kernel],
                    stride=self.low_conv_options["stride"][i_kernel],
                    padding=padding_same(
                        input=self.input_sz,
                        kernel=self.low_conv_options["kernel"][i_kernel],
                        stride=1,
                        dilation=1))
                layers += [
                    conv,
                    nn.BatchNorm1d(x),
                    # nn.ReLU(inplace=True)
                    nn.ELU(inplace=True, alpha=2.0)
                    ]
                in_chan = self.low_conv_hidden_dim = x
                self.input_sz /= self.low_conv_options["stride"][i_kernel]
                i_kernel += 1
            pass    # for
        return nn.Sequential(*layers)

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class ShortcutBlock(nn.Module):
    '''Pass a Sequence and add identity shortcut, following a ReLU.'''

    def __init__(
        self, layers=None, in_channels=None, out_channels=None,
        point_conv_group=1
    ):
        super(ShortcutBlock, self).__init__()
        self.iter = 0
        self.layers = layers
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, bias=False, groups=point_conv_group),
                nn.BatchNorm1d(out_channels)
            )
        self.act = nn.ELU(inplace=True, alpha=2.0)

    def forward(self, x):
        # self.debug(f'input: {x.shape}')

        out = self.layers(x)
        # self.debug(f'layers out: {out.shape}')

        out += self.shortcut(x)
        # self.debug(f'shortcut out: {out.shape}')

        # out = F.relu(out)
        out = self.act(out)

        self.iter += 1
        return out

    def debug(self, *args):
        if self.iter == 0:
            print(self.__class__.__name__, args)


if __name__ == '__main__':
    Hz = 100
    SEG_SEC = 3
    SEG_SIZE = Hz * SEG_SEC
    SEG_SLIDE_SEC = 1  # SEG_SEC
    LOW_CONV_STRIDE = 1
    IN_CHAN = 1
    OUT_CHAN = 24
    N_LAYERS_PER_BLOCK = 2
    N_BLOCK = 2
    NUM_CLASSES = 2
    model = ClassicConv(
        SEG_SIZE,
        shortcut_conn=True,
        pooling_off=True,
        in_channels=IN_CHAN,
        kernels=[5 for i in range(N_BLOCK)],
        out_channels=[OUT_CHAN for i in range(1, N_BLOCK+1)],
        conv_groups=[1 for i in range(N_BLOCK)],
        n_conv_layers_per_block=N_LAYERS_PER_BLOCK,
        n_blocks=N_BLOCK,
        n_classes=NUM_CLASSES,
        low_conv_options={
            "cfg": [OUT_CHAN],
            'stride': [LOW_CONV_STRIDE],
            "kernel": [11],
            'conv_groups': [1]},
        log=print
    )
    print(model)
    x = torch.rand(1, 1, 300)
    y = model(x)
