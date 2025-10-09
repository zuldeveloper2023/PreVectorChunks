# rlchunker/model.py
import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim + 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

def build_policy(embedding_dim=384, hidden_dim=128, device="cpu"):
    model = PolicyNetwork(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    return model.to(device)
