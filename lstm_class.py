import torch
from torch import nn

# DEFINE LSTM STRUCTURE:
class MyLSTM(nn.Module):
    # hidden_d - the size of the hidden LSTM layers
    # map_d - the flattened/encoded map dimension
    def __init__(self, hidden_d=120, map_d=128, device="cpu"):
        self.hidden_d = hidden_d
        self.device = device
        super(MyLSTM, self).__init__()

        # map hidden layer
        self.lstm_map = nn.LSTMCell(input_size=map_d, hidden_size=self.hidden_d, device=self.device)

        # points hidden layer
        self.lstm_points = nn.LSTMCell(input_size=4, hidden_size=self.hidden_d, device=self.device)
        
        # "upper" hidden layer
        self.lstm1 = nn.LSTMCell(input_size=self.hidden_d*2, hidden_size=self.hidden_d, device=self.device)
        self.fc = nn.Linear(self.hidden_d, 2, device=self.device)


    def forward(self, goal_point, current_point, map):

        # Creation of cell state and hidden state for map hidden layer
        hidden_state_map = torch.zeros(1, self.hidden_d).to(self.device)
        cell_state_map = torch.zeros(1, self.hidden_d).to(self.device)

        # Creation of cell state and hidden state for points hidden layer
        hidden_state_points = torch.zeros(1, self.hidden_d).to(self.device)
        cell_state_points = torch.zeros(1, self.hidden_d).to(self.device)

        # Creation of cell state and hidden state for "upper" hidden layer
        hidden_state_1 = torch.zeros(1, self.hidden_d).to(self.device)
        cell_state_1 = torch.zeros(1, self.hidden_d).to(self.device)

        outputs = []

        # initialize weights to random[-0.1, 0.1) (need to update initialzation to match paper)
        # weights initialization
        torch.nn.init.xavier_normal_(hidden_state_map)
        torch.nn.init.xavier_normal_(cell_state_map)

        torch.nn.init.xavier_normal_(hidden_state_points)
        torch.nn.init.xavier_normal_(cell_state_points)

        torch.nn.init.xavier_normal_(hidden_state_1)
        torch.nn.init.xavier_normal_(cell_state_1)

        # Concatenate start and goal
        points = torch.cat([current_point, goal_point], 0).unsqueeze(0)

        hidden_state_map, cell_state_map = self.lstm_map(map, (hidden_state_map, cell_state_map))
        hidden_state_points, cell_state_points = self.lstm_points(points, (hidden_state_points, cell_state_points))

        # Concatenate the output the lstm layer output from points and map into a single input to the final "upper" hidden layer
        final_layer_input = torch.cat([hidden_state_map, hidden_state_points], 1)
        hidden_state_1, cell_state_1 = self.lstm1(final_layer_input, (hidden_state_1, cell_state_1))
        
        # Last hidden state is passed through a fully connected neural net
        output = self.fc(hidden_state_1)	
        outputs.append(output)

        outputs = torch.cat(outputs, dim=0)
        
        return outputs