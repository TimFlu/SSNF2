import torch
import torch.nn as nn

# Simple NN with Linear functions and ReLU activation function
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.relu(self.fc(x))
        for layer in self.layers:
            x = self.relu(layer(x))
        x = torch.sigmoid(self.output_layer(x))
        return x


# ************** Build the Neural Network (used prior to implementing the above) ****************** #
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(14, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x
  