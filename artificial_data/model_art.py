import torch


# create a MLP with 10 layers and a head with the number of output recieve as parameter
class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(10):
            self.layers.append(torch.nn.Linear(input_dim, input_dim))
            self.layers.append(torch.nn.ReLU())
        self.head = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim // 2, output_dim)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)