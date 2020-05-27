#network.py
import torch
from torch_geometric.nn import GATConv
from utils.gin_conv2 import GINConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LeakyReLU
import torch.nn.functional as F


class GINN(torch.nn.Module):
    def __init__(self):
        super(GINN, self).__init__()
        
        input_ch = 64
        output_ch = int(input_ch/2)
        self.l1 = torch.nn.Linear(input_ch, output_ch)
        torch.nn.init.xavier_uniform_(self.l1.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        input_ch = output_ch
        output_ch = int(input_ch/2)
        self.l2 = torch.nn.Linear(input_ch, output_ch)
        torch.nn.init.xavier_uniform_(self.l2.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        return x

class NetGINConv(torch.nn.Module):
    def __init__(self, num_features, output_size):
        super(NetGINConv, self).__init__()
        self.num_cords = 2
        self.input_steps = int(num_features/self.num_cords)

        input_ch = 2
        output_ch = 64
        self.conv2Da = torch.nn.Conv2d(input_ch, output_ch, (2, 2),stride=2)
        torch.nn.init.xavier_uniform_(self.conv2Da.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))
        
        input_ch = output_ch
        output_ch = output_ch*2
        self.conv2Db = torch.nn.Conv2d(input_ch, output_ch, (2, 1), stride=2)
        torch.nn.init.xavier_uniform_(self.conv2Db.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        input_ch = output_ch
        output_ch = output_ch*2
        self.conv2Dc = torch.nn.Conv2d(input_ch, output_ch, (2, 1), stride=2)
        torch.nn.init.xavier_uniform_(self.conv2Dc.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        self.fc = torch.nn.Linear(int(num_features*2),int(num_features*4))
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        nn = GINN()
        nn2 = GINN()
        self.conv1 = GINConv(nn, nn2, train_eps=True)

        input_ch = output_ch
        output_ch = output_size
        self.conv2Dd = torch.nn.Conv2d(input_ch, output_ch, (1, 1))
        torch.nn.init.xavier_uniform_(self.conv2Dd.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

    def forward(self, x, x_real, edge_index):
        x1 = F.leaky_relu(self.fc(x_real))
        x1 = F.leaky_relu(self.conv1(x1, edge_index))
        x1 = x1.reshape(x.shape)
        x = torch.cat((x,x1),1)
        x = F.leaky_relu(self.conv2Da(x))
        x = F.leaky_relu(self.conv2Db(x))
        x = F.leaky_relu(self.conv2Dc(x))
        #Prediction
        x = F.leaky_relu(self.conv2Dd(x))
        return x