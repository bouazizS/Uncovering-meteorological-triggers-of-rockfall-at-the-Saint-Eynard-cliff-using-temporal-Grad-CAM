import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchview import draw_graph
from torchviz import make_dot


class CNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        """
        Args:
            input_shape (tuple): (sequence_length, num_channels)
            num_classes (int): number of output classes
        """
        super(CNN, self).__init__()
        seq_len, num_channels = input_shape

        # --- Convolutional layers ---
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # --- Compute flattened size dynamically ---
        with torch.no_grad():
            x = torch.zeros(1, num_channels, seq_len)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            flatten_dim = x.numel()

        # --- Fully connected layers ---
        self.fc1 = nn.Linear(flatten_dim, 128)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input shape: (batch, seq_len, channels)
        x = x.permute(0, 2, 1)  # -> (batch, channels, seq_len)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc_out(x)
        return x
    


class CNNBranch(nn.Module):
    """CNN branch configurable: number of conv layers, filters, and kernel size"""
    def __init__(self, input_shape, n_layers=3, base_filters=32, kernel_size=3):
        super(CNNBranch, self).__init__()
        seq_len, num_channels = input_shape

        layers = []
        in_channels = num_channels
        filters = base_filters

        # Dynamically build conv layers
        for i in range(n_layers):
            layers.append(nn.Conv1d(in_channels, filters, kernel_size=kernel_size, padding='same'))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = filters
            filters *= 2  # double filters each layer

        self.conv_stack = nn.Sequential(*layers)

        # compute flattened size
        with torch.no_grad():
            x = torch.zeros(1, num_channels, seq_len)
            x = self.conv_stack(x)
            self.flatten_dim = x.numel()

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        return x


class CNNLateFusion(nn.Module):
    def __init__(self, input_shape_1, input_shape_2, num_classes, 
                 n_layers=3, base_filters=32, kernel_size=3, dropout=0.3):
        super(CNNLateFusion, self).__init__()

        self.branch1 = CNNBranch(input_shape_1, n_layers, base_filters, kernel_size)
        self.branch2 = CNNBranch(input_shape_2, n_layers, base_filters, kernel_size)

        fusion_dim = self.branch1.flatten_dim + self.branch2.flatten_dim
        self.fc1 = nn.Linear(fusion_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x1, x2):
        f1 = self.branch1(x1)
        f2 = self.branch2(x2)
        fused = torch.cat([f1, f2], dim=1)
        x = F.relu(self.fc1(fused))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x

# if __name__ == "__main__":
batch_size = 32
seq_len_1, channels_1 = 336, 1
seq_len_2, channels_2 = 336, 1
num_classes = 2

model = CNNLateFusion(
    input_shape_1=(seq_len_1, channels_1),
    input_shape_2=(seq_len_2, channels_2),
    num_classes=num_classes
)

#     print(model)

#     model_graph = draw_graph( model, input_size=[(1, 1,336), (1, 1, 336)],  expand_nested=True)
#     model_graph = draw_graph( model, input_size=[(1, 1, 336), (1, 1, 336)],  expand_nested=True)

#     model_graph.visual_graph.render(filename="CNN_two branches_new", format="png")



total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total params: {total_params}")
print(f"Trainable params: {trainable_params}")