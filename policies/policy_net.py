import torch
from torch import nn
import torch.nn.functional as F


class PolicyNet(nn.Module):  # Network adapted from https://github.com/younik/breedgym-train

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnn = nn.Sequential(
             nn.Conv1d(2, 64, 32, 8),
            nn.ReLU(),
            nn.Conv1d(64, 4, 8, 4),
            nn.ReLU(),
            nn.Flatten()
        ).to(self.device)

        self.policy_head = nn.Linear(116, 32).to(self.device)
        # self.key_layer = nn.Linear(116, 32).to(self.device)
        # self.query_layer = nn.Linear(116, 32).to(self.device)

    def forward(self, x):
        batch_pop = x.reshape(-1, x.shape[-2], x.shape[-1])
        batch_pop = batch_pop.permute(0, 2, 1)
        batch_pop = batch_pop.type(torch.float32)
        cnn_out1 = self.cnn(batch_pop)

        # make invariant to diploidy
        chan_indices = torch.arange(x.shape[-1])
        chan_indices[0] = 1
        chan_indices[1] = 0
        cnn_out2 = self.cnn(batch_pop[:, chan_indices])
        features = cnn_out1 + cnn_out2
        features = features.reshape(*x.shape[:-2], -1)

        features = self.policy_head(features)
        out = -F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(2), dim=-1)

        # transformed_key = self.key_layer(features)
        # transformed_query = self.query_layer(features)
        # scores = torch.matmul(transformed_query, transformed_key.transpose(-2, -1)) / 5.65 # sqrt 32

        # # # Apply softmax to get attention weights
        # out = F.softmax(scores.flatten(start_dim=-2), dim=-1) * 2
        # out = out.reshape(scores.shape)
        return out