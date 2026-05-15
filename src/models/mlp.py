import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128], num_classes=10, dropout=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        
        layers = []
        in_features = input_size
        # All layers except last one
        for hidden_size in hidden_sizes[:-1]:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_features = hidden_size
        
        # Last hidden layer without dropout
        layers.extend([
            nn.Linear(in_features, hidden_sizes[-1]),
            nn.ReLU()
        ])
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)