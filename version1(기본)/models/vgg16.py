import torch.nn as nn


def calc_n(input_shape, channels):
    w, h = input_shape
    w /= 2**len(channels)
    h /= 2**len(channels)

    return int(w*h*channels[-1])

class VGG16(nn.Module):
    def __init__(self, 
                 input_shape: tuple, 
                 channels: list, 
                 dense_dims: list, 
                 n_class: int,
                 dropout_rate: int = 0.1,
                 ):
        super(VGG16, self).__init__()
        self.input_shape = input_shape
        self.channels = channels
        self.dense_dims = dense_dims
        self.n_class = n_class
        self.dropout_rate = dropout_rate

        self.conv_layers = self.constrcut_convs()
        self.flatten = nn.Flatten()

        self.dense = self.construct_denses()
        self.cls = nn.Linear(self.dense_dims[-1], n_class)
        self.dropout = nn.Dropout(self.dropout_rate)

    def construct_denses(self):
        in_dim = calc_n(self.input_shape, self.channels)
        dense_list = []
        for dense_dim in self.dense_dims:
            dense_list.append(
                nn.Linear(in_dim, dense_dim)
            )
            dense_list.append(
                nn.ReLU()
            )
            in_dim = dense_dim
        return nn.Sequential(*dense_list)

    def constrcut_convs(self):
        conv_list = []
        in_channel = 3
        for i, out_channel in enumerate(self.channels):
            if i == 0:
                n_conv = 2
            else:
                n_conv = 3
            conv_list.append(
                self.conv_layer(in_channel, out_channel, n_conv)
            )
            conv_list.append(
                nn.MaxPool2d(2, 2)
            )
            in_channel = out_channel
        return nn.Sequential(*conv_list)

    def conv_layer(self, in_channel, out_channel, n_conv=3):
        conv_list = []
        for _ in range(n_conv):
            conv_list.append(
                nn.Conv2d(in_channel, out_channel, (3, 3), stride=1, padding=1),
            )
            in_channel = out_channel
            conv_list.append(
                nn.ReLU()
            )
        return nn.Sequential(*conv_list)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.dropout(x)

        x = self.dense(x)
        x = self.dropout(x)
        
        x = self.cls(x)

        return x
