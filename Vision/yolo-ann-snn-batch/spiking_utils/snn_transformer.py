from spiking_utils import ann_parser
from spiking_utils.spike_layer import *


def is_layer_weighted_spike(layer):
    return isinstance(layer, SpikeConv2d) or isinstance(layer, SpikeConvTranspose2d) or isinstance(layer, SpikeLinear)


class DataStatus:
    def __init__(self, max_num=1e7, channel_wise=True):
        self.pool = []
        self.num = 0
        self.max_num = max_num
        self.channel_wise = channel_wise

    def append(self, data):
        if self.channel_wise:
            b, c = data.size()[:2]
            self.pool.append(data.transpose(0, 1).contiguous().view(c, -1))
            self.num += self.pool[-1].size()[0] * self.pool[-1].size()[1]
        else:
            self.pool.append(data.view(-1))
            self.num += self.pool[-1].size()[0]
        if self.num > self.max_num:
            self.random_shrink()

    def random_shrink(self):
        if self.channel_wise:
            tensor = torch.cat(self.pool, 1)
            c, n = tensor.size()[:2]
            tensor = tensor[:, torch.randint(n, size=[int(n // 2)])]
        else:
            tensor = torch.cat(self.pool, 0)
            tensor = tensor[torch.randint(
                len(tensor), size=[int(self.max_num // 2)])]
        self.pool.clear()
        self.pool.append(tensor)

    def fraction_max(self, fraction=1, relu=True, max_num=1e6):
        if self.channel_wise:
            tensor = torch.cat(self.pool, 1)  # shape [n_channels, n]
        else:
            tensor = torch.cat(self.pool, 0)  # shape [n]
        if relu:
            tensor = F.relu(tensor)
        if self.channel_wise:
            tensor_sort = tensor.sort(1)[0]
            return tensor_sort[:, int(fraction * tensor_sort.size(1))]
        else:
            tensor_sort = tensor.sort()[0]
            return tensor_sort[int(fraction * tensor_sort.size(0))]


class SNNTransformer:
    def __init__(self, args, net, device):
        self.original_net = net
        self.timesteps = args.timesteps
        self.device = device
        self.snn_dag = None
        self.ann_snn_layer_mapping = {}
        self.reset_mode = args.reset_mode
        self.layer2name = {}
        self.input_status = {}
        self.output_status = {}
        self.input_generator = {}
        self.channel_wise = args.channel_wise
# 这个函数的主要目的是将一个传统的人工神经网络模型转化为一个脉冲神经网络的DAG结构，
# 并为其中的某些特定层（可能是加权的脉冲层）注册钩子，以便在训练过程中跟踪它们的输入和输出状态。
    def init_dag(self, inputs):
        # 暂时还没看懂这个，全是建容器的操作
        self.snn_dag = ann_parser.parse_ann_model(self.original_net, inputs)
        self.snn_dag.to(self.device)
        # trace spike layers
        # 这是一个循环，它遍历snn_dag中的所有模块。对于每个模块，我们得到了它的名字layer_name和它本身layer。
        for layer_name, layer in self.snn_dag.named_modules():
            # 这里建立了一个从层对象到其名字的映射
            self.layer2name[layer] = layer_name
            # 它使用is_layer_weighted_spike函数来检查当前层是否是一个“加权的脉冲层”。如果是，接下来的代码块会执行。

            # 为当前层的输入和输出状态分别创建DataStatus对象。DataStatus可能是一个用于跟踪数据状态（例如最大值、最小值等）的类。
            if is_layer_weighted_spike(layer):
                self.input_status[layer_name] = DataStatus(
                    channel_wise=self.channel_wise)
                self.output_status[layer_name] = DataStatus(
                    channel_wise=self.channel_wise)
                # 它将在每次前向传播时被调用。这个钩子的目的是跟踪并存储当前层的输入和输出数据。
                def forward_hook(m, inputs, outputs):
                    self.input_status[self.layer2name[m]].append(
                        inputs[0].detach().cpu())
                    self.output_status[self.layer2name[m]].append(
                        outputs.detach().cpu())

                layer.register_forward_hook(forward_hook)

    def inference_get_status(self, train_loader, num_iters):
        for batch_i, (imgs, targets, paths, _) in enumerate(train_loader):
            if batch_i > num_iters:
                break
            data = imgs.to(self.device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            if self.snn_dag is None:
                self.init_dag([data])
            out = self.snn_dag(data)
        # freeze hook of spike layers
        # 清除了所有的前向钩子
        for layer_name, layer in self.snn_dag.named_modules():
            layer._forward_hooks.clear()

    def fuse_bn(self):
        for layer_name, layer in self.snn_dag.named_modules():
            if (isinstance(layer, SpikeConv2d) or isinstance(layer, SpikeConvTranspose2d)) and layer.bn is not None:
                layer.weight.data[...] = layer.weight * (
                        layer.bn.weight / torch.sqrt(layer.bn.running_var + layer.bn.eps)).view(-1, 1, 1, 1)
                if layer.bias is not None:
                    layer.bias.data[...] = (layer.bias - layer.bn.running_mean) * layer.bn.weight / torch.sqrt(
                        layer.bn.running_var + layer.bn.eps) + layer.bn.bias
                else:
                    bias = (-layer.bn.running_mean) * layer.bn.weight / torch.sqrt(
                        layer.bn.running_var + layer.bn.eps) + layer.bn.bias
                    bias = nn.Parameter(bias)
                    layer._parameters['bias'] = bias
                    layer.bias = bias
                layer.bn = None
                print(f"Fuse the weights in {layer_name}")

    def gen_weight(self, layer, max_in, max_out):
        weight = layer.weight
        if len(weight.size()) == 4:
            scale_snn = max_in.view(1, -1, 1, 1) / max_out.view(-1, 1, 1, 1)
        elif len(weight.size()) == 2:
            scale_snn = max_in.view(1, -1) / max_out.view(-1, 1)
        else:
            raise NotImplementedError
        return weight.data * scale_snn

    def gen_bias(self, layer, max_out):
        return layer.bias.data / max_out

    def gen_Vthr(self, layer):
        return 1

    def generate_snn(self):
        # 这通常是为了将网络中的卷积层和批量归一化层融合为一个层，以简化和优化网络。
        self.fuse_bn()
        # 上面把每一层都转换的加进来了
        for layer_i, (layer_name, layer) in enumerate(self.snn_dag.named_modules()):
            # 检查当前层是否是一个加权的脉冲层
            if is_layer_weighted_spike(layer):
                print(f"processing layer {layer_name}")
                # TODO: supporting specify the first layer for multi input branch network
                # 从存储的状态字典中获取该层的输入和输出状态。
                input_status = self.input_status[layer_name]
                output_status = self.output_status[layer_name]
                # maxin 和 maxout我就步多说了
                max_in = input_status.fraction_max(fraction=0.99999).to(self.device)
                max_out = output_status.fraction_max(fraction=0.99999).to(self.device)
                # 权重转换
                if layer_i == 0:
                    layer.weight.data[...] = self.gen_weight(
                        layer, torch.ones(1).to(self.device), max_out)
                else:
                    layer.weight.data[...] = self.gen_weight(
                        layer, max_in, max_out)
                # 为当前层生成一个阈值Vthr。
                layer.Vthr[...] = self.gen_Vthr(layer)
                # 设置层的输出缩放因子为max_out。 估计就是记录一下，没看到哪里用他
                layer.out_scales.data[...] = max_out
                # 如果该层有偏置，生成一个新的偏置，并将其设置为max_out的值。同时，将此偏置值设置为层的"leakage"属性。
                if layer.bias is not None:
                    new_bias = self.gen_bias(layer, max_out)
                    layer.bias.data[...] = new_bias
                    layer.leakage = layer.bias.data
                print(f"set {layer_name}: Vthr {layer.Vthr}")
        # unwrap the layers
        # 清除SNN DAG中所有加权脉冲层的前向钩子。这可能是为了确保在转换后，这些层不执行任何额外的操作。
        for layer in self.snn_dag.modules():
            if is_layer_weighted_spike(layer):
                layer._forward_hooks.clear()
        # 为SNN DAG中的所有模块设置reset_mode属性。
        for m in self.snn_dag.modules():
            m.reset_mode = self.reset_mode
        print(f"Transfer ANN to SNN Finished")
        return self.snn_dag
