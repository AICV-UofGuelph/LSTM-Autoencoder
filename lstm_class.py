import torch
from torch import nn

# DEFINE LSTM STRUCTURE:
class LSTM_Single_Layer(nn.Module):
    # hidden_d - the size of the hidden LSTM layers
    # map_d - the flattened/encoded map dimension
    def __init__(self, hidden_d=120, map_d=128, device="cpu"):
        self.hidden_d = hidden_d
        self.device = device
        super(LSTM_Single_Layer, self).__init__()

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

# DEFINE LSTM STRUCTURE (2 Layer):
class LSTM_Two_Layer(nn.Module):
  # hidden_d - the size of the hidden LSTM layers
  # map_d - the flattened/encoded map dimension
  def __init__(self, hidden_d=120, map_d=128, device="cpu", DROPOUT_P=0.5, DROPOUT = [1, 1]):
    self.hidden_d = hidden_d
    self.device = device
    super(LSTM_Two_Layer, self).__init__()

    # hidden layer - encoded map
    self.lstm_map = nn.LSTMCell(input_size=map_d, hidden_size=self.hidden_d, device=self.device)

    # hidden layer - current and goal point
    self.lstm_points = nn.LSTMCell(input_size=4, hidden_size=self.hidden_d, device=self.device)
    
    # hidden layer - concat #1
    self.lstm1 = nn.LSTMCell(input_size=self.hidden_d*2, hidden_size=self.hidden_d*2, device=self.device)

    # hidden layer - concat #2
    self.lstm2 = nn.LSTMCell(input_size=self.hidden_d*2, hidden_size=self.hidden_d, device=self.device)

    self.fc = nn.Linear(self.hidden_d, 2, device=self.device)

    #dropout for lstm layers 1 and 2
    self.do1 = nn.Dropout(p=DROPOUT_P)
    self.do2 = nn.Dropout(p=DROPOUT_P)

    self.hidden_states = self.init_hidden() # creates a list of empty tensors for cell and hidden states

  def init_hidden(self):
    return(
      [torch.zeros(1, self.hidden_d).to(self.device), # hidden states - encoded map
      torch.zeros(1, self.hidden_d).to(self.device), # cell states - encoded map
      torch.zeros(1, self.hidden_d).to(self.device), # hidden states - current and goal point
      torch.zeros(1, self.hidden_d).to(self.device), # cell states - current and goal point
      torch.zeros(1, self.hidden_d*2).to(self.device), # hidden states - concat layer
      torch.zeros(1, self.hidden_d*2).to(self.device), # cell states - concat layer
      torch.zeros(1, self.hidden_d).to(self.device), # hidden states - concat layer#2
      torch.zeros(1, self.hidden_d).to(self.device)] # cell states - concat layer#2
    )

  def forward(self, goal_point, current_point, map):

    encoded_map_hidden_state = self.hidden_states[0].clone().detach() # clone copies of hidden and cell state maps
    encoded_map_cell_state = self.hidden_states[1].clone().detach()
    points_hidden_state = self.hidden_states[2].clone().detach()
    points_cell_state = self.hidden_states[3].clone().detach()
    hidden_state_1 = self.hidden_states[4].clone().detach()
    cell_state_1 = self.hidden_states[5].clone().detach()
    hidden_state_2 = self.hidden_states[6].clone().detach()
    cell_state_2 = self.hidden_states[7].clone().detach()

    outputs = [] # create an empty list to hold network output

    # Concatenate start and goal
    points = torch.cat([current_point, goal_point], 0).unsqueeze(0)

    # run the encoded map and concatenated points through the first network layers
    encoded_map_hidden_state, encoded_map_cell_state = self.lstm_map(map, (encoded_map_hidden_state, encoded_map_cell_state))
    points_hidden_state, points_cell_state = self.lstm_points(points, (points_hidden_state, points_cell_state))

    # Concatenate the output from the lstm layer and output from points layer into a single input to the final hidden layer
    # (note that for LSTMCell layer, the output is the "hidden state map")
    final_layer_input = torch.cat([encoded_map_hidden_state, points_hidden_state], 1)

    # (Optional) add dropout before lstm layer 1
    if DROPOUT[0]: final_layer_input = self.do1(final_layer_input) # dropout here
    # feed thorugh the first concat layer
    hidden_state_1, cell_state_1 = self.lstm1(final_layer_input, (hidden_state_1, cell_state_1))

    # (Optional) add dropout before lstm layer 2
    if DROPOUT[1]: hidden_state_1 = self.do2(hidden_state_1) # dropout here
    # feed thorugh the second concat layer
    hidden_state_2, cell_state_2 = self.lstm2(hidden_state_1, (hidden_state_2, cell_state_2))
      
    # Last hidden state is passed through a fully connected neural net to get down to two in size (output is next point predicted)
    output = self.fc(hidden_state_2)	
    outputs.append(output)

    outputs = torch.cat(outputs, dim=0)

    # save the updated cell and hidden state maps
    self.hidden_states[0] = encoded_map_hidden_state
    self.hidden_states[1] = encoded_map_cell_state
    self.hidden_states[2] = points_hidden_state
    self.hidden_states[3] = points_cell_state
    self.hidden_states[4] = hidden_state_1
    self.hidden_states[5] = cell_state_1
    self.hidden_states[6] = hidden_state_2
    self.hidden_states[7] = cell_state_2
    
    return outputs