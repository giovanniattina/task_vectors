import torch

# create a MLP with 10 layers and a head with the number of output receive as parameter
class MLPBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=10):
        super(MLPBlock, self).__init__()
        
        layers = []
        for i in range(num_layers):
            layers.extend([
                torch.nn.Linear(input_dim, output_dim),
                torch.nn.ReLU()
            ])
        
        self.block = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)
    

class BackBone(torch.nn.Module):
    def __init__(self, input_dim):
        super(BackBone, self).__init__()
 
        self.input_dim = input_dim
        self.output_dim = input_dim // 2
        
        # First block: input_dim -> input_dim
        self.block1 = MLPBlock(input_dim, input_dim, num_layers=5)
        
        # Transition layer
        self.transition = torch.nn.Linear(input_dim, self.output_dim)
        
        # Second block: hidden_dim -> hidden_dim
        self.block2 = MLPBlock(self.output_dim, self.output_dim, num_layers=5)

    def forward(self, x):
        x = self.block1(x)
        x = self.transition(x)
        x = self.block2(x)
        return x

    def get_output_dim(self):
        return self.output_dim


class Head(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Head, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.input_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.input_dim // 2, self.output_dim)
        )
    
    def forward(self, x):
        return self.head(x)


class ClassificationModel(torch.nn.Module):
    def __init__(self, backbone, head):
        super(ClassificationModel, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        outputs = self.head(features)
        return outputs
