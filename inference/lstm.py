from torch import nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, classes, leaky_relu=False):
        super(SimpleLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.3)
        self.fc1 = nn.Linear(hidden_dim, classes)
        self.activation = nn.LeakyReLU() if leaky_relu else nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = x[:, -1, :]
        x = self.activation(x)
        x = self.fc1(x)
        out = self.softmax(x)
        return out