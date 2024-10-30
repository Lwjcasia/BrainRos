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
    """ Spiking Actor Network with Attention and Feature Fusion """
    def __init__(self, state_num, action_num, lidar_dim, device, batch_window=50, hidden1=256, hidden2=256, hidden3=256):
        super(ActorNetSpiking, self).__init__()
        self.state_num = state_num  # 用于线速度、角速度、目标距离、方向
        self.lidar_dim = lidar_dim  # 雷达数据维度
        self.action_num = action_num
        self.device = device
        self.batch_window = batch_window
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3

        # 特征提取模块
        self.lidar_net = nn.Sequential(
            nn.Linear(self.lidar_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU()
        )
        
        self.state_net = nn.Sequential(
            nn.Linear(self.state_num, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU()
        )

        # 注意力机制
        self.attention_fc = nn.Linear(hidden2 * 2, hidden2)
        self.softmax = nn.Softmax(dim=-1)
        
        # 脉冲网络层
        self.fc1 = nn.Linear(hidden2 * 2, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, self.action_num)

    def neuron_model(self, syn_func, pre_layer_output, current, volt, spike):
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt * NEURON_VDECAY * (1. - spike) + current
        spike = self.pseudo_spike(volt)
        return current, volt, spike

    def forward(self, x_state, x_lidar, batch_size):
        # 特征提取
        lidar_features = self.lidar_net(x_lidar)
        state_features = self.state_net(x_state)

        # 特征融合
        combined_features = torch.cat([state_features, lidar_features], dim=1)

        # 注意力机制
        attention_weights = self.softmax(self.attention_fc(combined_features))
        combined_features = combined_features * attention_weights
        
        # 脉冲网络传播
        fc1_u = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc1_v = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc1_s = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc2_u = torch.zeros(batch_size, self.hidden2, device=self.device)
        fc2_v = torch.zeros(batch_size, self.hidden2, device=self.device)
        fc2_s = torch.zeros(batch_size, self.hidden2, device=self.device)
        fc3_u = torch.zeros(batch_size, self.hidden3, device=self.device)
        fc3_v = torch.zeros(batch_size, self.hidden3, device=self.device)
        fc3_s = torch.zeros(batch_size, self.hidden3, device=self.device)
        fc4_u = torch.zeros(batch_size, self.action_num, device=self.device)
        fc4_v = torch.zeros(batch_size, self.action_num, device=self.device)
        fc4_s = torch.zeros(batch_size, self.action_num, device=self.device)
        fc4_sumspike = torch.zeros(batch_size, self.action_num, device=self.device)

        for step in range(self.batch_window):
            fc1_u, fc1_v, fc1_s = self.neuron_model(self.fc1, combined_features, fc1_u, fc1_v, fc1_s)
            fc2_u, fc2_v, fc2_s = self.neuron_model(self.fc2, fc1_s, fc2_u, fc2_v, fc2_s)
            fc3_u, fc3_v, fc3_s = self.neuron_model(self.fc3, fc2_s, fc3_u, fc3_v, fc3_s)
            fc4_u, fc4_v, fc4_s = self.neuron_model(self.fc4, fc3_s, fc4_u, fc4_v, fc4_s)
            fc4_sumspike += fc4_s
        
        out = fc4_sumspike / self.batch_window
        return out

