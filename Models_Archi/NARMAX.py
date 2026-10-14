import torch
import torch.nn as nn
from torchsummary import summary
import torchvision
from torchview import draw_graph
from torchviz import make_dot


class NARX_MLP(nn.Module):
    def __init__(self, input_size, hidden_size=128, n_lags=14):
        super().__init__()
        
        self.n_lags = n_lags
        self.fc1 = nn.Linear((input_size + 1) * n_lags, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y_lags):
        """
        x: [batch, lags, features]
        y_lags: [batch, lags]
        """
        
        x = x.reshape(x.size(0), -1)
        y_lags = y_lags.reshape(y_lags.size(0), -1)
        
        out = torch.cat([x, y_lags], dim=1)
        
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.out(out))
        
        return out


import torch
import torch.nn as nn
import numpy as np


class NARXModel(nn.Module):
    """
    Neural NARX with explicit lags.

    Paramètres
    ----------
    n_lags_y    : nombre de lags sur la sortie y (partie AR)
    n_lags_x    : nombre de lags sur x1 et x2   (partie X)
    num_classes : nombre de classes (2 pour binaire)
    hidden_dims : liste des tailles des couches cachées du MLP
    dropout     : dropout entre les couches
    """

    def __init__(
        self,
        n_lags_y=24,
        n_lags_x=168,
        num_classes=2,
        hidden_dims=[128, 64],
        dropout=0.3,
    ):
        super().__init__()
        self.n_lags_y = n_lags_y
        self.n_lags_x = n_lags_x

        # taille du vecteur d'entrée Φ(t) = [y_lags | x1_lags | x2_lags]
        input_dim = n_lags_y + n_lags_x + n_lags_x

        # MLP : F(Φ(t)) non-linéaire
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def _extract_lags(self, x, n_lags):
        """
        x      : (batch, 1, T)  — série temporelle brute
        retour : (batch, n_lags) — les n_lags dernières valeurs
        """
        x_flat = x.squeeze(1)           # (batch, T)
        return x_flat[:, -n_lags:]      # (batch, n_lags)

    def forward(self, x1, x2, y_past):
        """
        x1, x2 : (batch, 1, 336)
        y_past  : (batch, n_lags_y)  — lags de y fournis explicitement
        """
        lags_x1 = self._extract_lags(x1, self.n_lags_x)   # (batch, n_lags_x)
        lags_x2 = self._extract_lags(x2, self.n_lags_x)   # (batch, n_lags_x)

        # vecteur NARX : Φ(t) = [y(t-1)..y(t-n) | x1_lags | x2_lags]
        phi = torch.cat([y_past, lags_x1, lags_x2], dim=1)  # (batch, input_dim)

        return self.mlp(phi)    # (batch, num_classes)



def build_y_lags(y_sequence, n_lags, device='cpu'):
    """
    Construit la matrice des lags de y pour tous les échantillons.

    y_sequence : tensor (N,) — labels encodés de tous les échantillons
                 dans l'ordre temporel
    n_lags     : int — nombre de lags à extraire
    retour     : tensor (N, n_lags) — pour chaque échantillon i,
                 contient [y(i-1), y(i-2), ..., y(i-n_lags)]
                 rempli de 0 pour les premiers échantillons (padding causal)
    """
    N = len(y_sequence)
    y_lags = torch.zeros(N, n_lags, dtype=torch.float32)

    for i in range(N):
        for lag in range(1, n_lags + 1):
            if i - lag >= 0:
                y_lags[i, lag - 1] = float(y_sequence[i - lag])
            # sinon reste 0 (padding causal)

    return y_lags.to(device)


# -----------------------------------------------------------------------
# Dataset adapté au NARX (inclut y_lags)
# -----------------------------------------------------------------------

class NARXDataset(torch.utils.data.Dataset):
    """
    Dataset pour NARX : retourne (x1, x2, y_lags, y_target).
    """

    def __init__(self, x1, x2, y, y_lags):
        self.x1     = x1
        self.x2     = x2
        self.y      = y
        self.y_lags = y_lags

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.x1[idx], self.x2[idx], self.y_lags[idx]), self.y[idx]
    


model = NARXModel(
    n_lags_y=24,
    n_lags_x=168,
    num_classes=2,
    hidden_dims=[128, 64],
    dropout=0.3,
)

batch_size = 1

x1 = torch.randn(batch_size, 1, 336)
x2 = torch.randn(batch_size, 1, 336)
y_past = torch.randn(batch_size, model.n_lags_y)

# draw graph
model_graph = draw_graph(
    model,                          # Le modèle à visualiser
    input_data=(x1, x2, y_past),   # Les entrées factices
    graph_name="NARX_Model",       # Nom du graphe
    depth=3,                       # Profondeur de visualisation
    expand_nested=True    # Dossier de sauvegarde
)
model_graph.visual_graph.render(filename="NARMAX", format="png")


total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total params: {total_params}")
print(f"Trainable params: {trainable_params}")


