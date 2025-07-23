import torch
import torch.nn as nn


NEURON_VTH = 0.5
NEURON_CDECAY = 1 / 2
NEURON_VDECAY = 3 / 4
SPIKE_PSEUDO_GRAD_WINDOW = 0.5


class PseudoSpikeRect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()


class ActorNetSpiking(nn.Module):
    """ Spiking Actor Network """
    def __init__(self, state_num, action_num, device, batch_window=50, hidden1=256, hidden2=256, hidden3=256):
        """

        :param state_num: number of states
        :param action_num: number of actions
        :param device: device used
        :param batch_window: window steps
        :param hidden1: hidden layer 1 dimension
        :param hidden2: hidden layer 2 dimension
        :param hidden3: hidden layer 3 dimension
        """
        super(ActorNetSpiking, self).__init__()
        self.state_num = state_num
        self.action_num = action_num
        self.device = device
        self.batch_window = batch_window
        self.hidden_dims = [hidden1, hidden2, hidden3]
        self.pseudo_spike = PseudoSpikeRect.apply

        layer_dims = [self.state_num] + self.hidden_dims + [self.action_num]
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=True))

    def neuron_model(self, syn_func, pre_layer_output, current, volt, spike):
        """
        Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt * NEURON_VDECAY * (1. - spike) + current
        spike = self.pseudo_spike(volt)
        return current, volt, spike

    def forward(self, x, batch_size):
        """

        :param x: state batch
        :param batch_size: size of batch
        :return: out
        """
        layer_dims = self.hidden_dims + [self.action_num]
        currents = [torch.zeros(batch_size, dim, device=self.device) for dim in layer_dims]
        voltages = [torch.zeros(batch_size, dim, device=self.device) for dim in layer_dims]
        spikes = [torch.zeros(batch_size, dim, device=self.device) for dim in layer_dims]
        
        output_sum_spikes = torch.zeros(batch_size, self.action_num, device=self.device)

        for step in range(self.batch_window):
            input_spike = x[:, :, step]
            
            layer_input = input_spike
            for i in range(len(self.layers)):
                currents[i], voltages[i], spikes[i] = self.neuron_model(
                    self.layers[i], layer_input, currents[i], voltages[i], spikes[i]
                )
                layer_input = spikes[i]

            output_sum_spikes += spikes[-1]

        out = output_sum_spikes / self.batch_window
        return out

