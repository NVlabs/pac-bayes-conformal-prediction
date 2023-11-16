from torch import nn

activation_aliases = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, n_hidden: int,
                 layer_width: int, activation: str) -> None:
        super().__init__()
        assert n_hidden >= 1
        
        layers = []
        layers.append(nn.Linear(input_dim, layer_width))
        layers.append(activation_aliases[activation]())
        for _ in range(n_hidden-1):
            layers.append(nn.Linear(layer_width,layer_width))
            layers.append(activation_aliases[activation]())
        layers.append(nn.Linear(layer_width,output_dim))
        
        self.backbone = nn.Sequential(*layers[:0])
        self.head = nn.Sequential(*layers[0:])

    def forward(self,inputs):
        features = self.backbone(inputs)
        return self.head(features)

        
        