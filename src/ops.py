import neuralnet_pytorch as nnt


class GraphXConv(nnt.Sequential):
    def __init__(self, input_shape, out_features, num_instances=None, bias=True, activation=None, weights_init=None, bias_init=None):
        super().__init__(input_shape=input_shape)
        self.out_features = out_features
        self.num_instances = num_instances if num_instances else input_shape[1]
        self.activation = activation
        pattern = list(range(len(input_shape)))
        pattern[-1], pattern[-2] = pattern[-2], pattern[-1]

        self.add_module('dimshuffle1', nnt.DimShuffle(pattern, input_shape=self.output_shape))
        self.add_module('conv_l', nnt.FC(self.output_shape, num_instances, bias=bias, activation=None,
                                         weights_init=weights_init, bias_init=bias_init))
        self.add_module('dimshuffle2', nnt.DimShuffle(pattern, input_shape=self.output_shape))
        self.add_module('conv_r', nnt.FC(self.output_shape, out_features, bias=bias, activation=activation))


class ResGraphXConv(nnt.Module):
    def __init__(self, input_shape, out_features, num_instances=None, bias=True, activation=None):
        super().__init__(input_shape)
        self.out_features = out_features
        self.activation = nnt.utils.function(activation)

        self.main = nnt.Sequential(input_shape=input_shape)
        self.main.add_module('conv_r_1', nnt.FC(self.main.output_shape, out_features, bias=bias, activation=activation))
        self.main.add_module('convr_2', GraphXConv(self.main.output_shape, out_features, num_instances, bias, None))
        if num_instances is None:
            self.res = (lambda x: x) if (out_features == input_shape[-1]) \
                else nnt.FC(input_shape, out_features, bias=bias, activation=None)
        else:
            self.res = GraphXConv(input_shape, out_features, num_instances, activation=None)

    def forward(self, input):
        res = self.res(input)
        output = self.main(input) + res
        return self.activation(output)

    @property
    @nnt.utils.validate
    def output_shape(self):
        return self.main.output_shape


class ResFC(ResGraphXConv):
    def __init__(self, input_shape, out_features, bias=True, activation=None):
        super().__init__(input_shape, out_features, bias=bias, activation=activation)
        self.main = nnt.Sequential(input_shape=input_shape)
        self.main.add_module('fc1', nnt.FC(self.main.output_shape, out_features, activation=activation))
        self.main.add_module('fc2', nnt.FC(self.main.output_shape, out_features, activation=None))


class LowRankGraphXConv(nnt.Sequential):
    def __init__(self, input_shape, out_features, num_instances=None, rank=None, bias=True, activation=None,
                 weights_init=None, bias_init=None):
        super().__init__(input_shape=input_shape)
        self.out_features = out_features
        self.num_instances = num_instances if num_instances else input_shape[1]
        self.rank = rank if rank is not None else self.num_instances // 2
        assert self.rank < self.num_instances, 'rank should be smaller than num_instances'

        self.activation = activation
        pattern = list(range(len(input_shape)))
        pattern[-1], pattern[-2] = pattern[-2], pattern[-1]

        self.add_module('dimshuffle1', nnt.DimShuffle(pattern, input_shape=self.output_shape))
        self.add_module('conv_l1', nnt.FC(self.output_shape, self.rank, bias=False, activation=None,
                                          weights_init=weights_init))
        self.add_module('conv_l2', nnt.FC(self.output_shape, self.num_instances, bias=bias, activation=None,
                                          weights_init=weights_init, bias_init=bias_init))
        self.add_module('dimshuffle2', nnt.DimShuffle(pattern, input_shape=self.output_shape))
        self.add_module('conv_r', nnt.FC(self.output_shape, out_features, bias=bias, activation=activation))


class ResLowRankGraphXConv(nnt.Module):
    def __init__(self, input_shape, out_features, num_instances=None, rank=None, bias=True, activation=None):
        super().__init__(input_shape)
        self.out_features = out_features
        self.activation = nnt.utils.function(activation)

        self.main = nnt.Sequential(input_shape=input_shape)
        self.main.add_module('conv_r_1', nnt.FC(self.main.output_shape, out_features, bias=bias, activation=activation))
        self.main.add_module('convr_2', LowRankGraphXConv(self.main.output_shape, out_features, num_instances, rank,
                                                          bias, None))
        if num_instances is None:
            self.res = (lambda x: x) if (out_features == input_shape[-1]) \
                else nnt.FC(input_shape, out_features, bias=bias, activation=None)
        else:
            self.res = LowRankGraphXConv(input_shape, out_features, num_instances, rank, activation=None)

    def forward(self, input):
        res = self.res(input)
        output = self.main(input) + res
        return self.activation(output)

    @property
    @nnt.utils.validate
    def output_shape(self):
        return self.main.output_shape
