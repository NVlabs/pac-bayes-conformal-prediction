from torch import nn

class LeNet(nn.Module):  
    """The LeNet-5 model."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1,6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6,16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(400,120), nn.Sigmoid(),
            nn.Linear(120,84), nn.Sigmoid(),
            nn.Linear(84,num_classes)
        )
    def forward(self,inputs):
        features = self.backbone(inputs)
        return self.head(features)