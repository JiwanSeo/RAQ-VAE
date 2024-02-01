import torch
from torch import nn
import torch.nn.functional as F


class EMAQuantizer(nn.Module):
        def __init__(self, embedding_dim, n_embed, decay=0.99, eps=1e-5):
                super().__init__()

                self.embedding_dim = embedding_dim
                self.n_embed = n_embed
                self.decay = decay
                self.eps = eps
                self.embed = nn.Embedding(self.n_embed, self.embedding_dim)

                self.register_buffer("cluster_size", torch.zeros(self.n_embed))
                self.register_buffer("embed_avg", self.embed.weight.t().clone())

        def forward(self, z_e):
                flatten = z_e.reshape(-1, self.embedding_dim)

                dist = (
                        flatten.pow(2).sum(1, keepdim=True)
                        - 2 * flatten @ self.embed.weight.t()
                        + self.embed.weight.pow(2).sum(1, keepdim=True).t()
                )
                _, embed_ind = (-dist).max(1)
                embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
                embed_ind = embed_ind.view(*z_e.shape[:-1])
                z_q = self.embed_code(embed_ind)

                if self.training:
                        embed_onehot_sum = embed_onehot.sum(0)
                        embed_sum = flatten.transpose(0, 1) @ embed_onehot

                        self.cluster_size.data.mul_(self.decay).add_(
                                embed_onehot_sum, alpha=1 - self.decay
                        )
                        self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1-self.decay)
                        n = self.cluster_size.sum()
                        cluster_size = (
                                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
                        )
                        embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
                        self.embed.weight.data.copy_(embed_normalized.t())

                commitment_cost = 0.25
                diff = commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()
                z_q = z_e + (z_q - z_e).detach()
                return z_q, diff, embed_ind

        def embed_code(self, embed_id):
                return F.embedding(embed_id, self.embed.weight)


class Quantizer(nn.Module):
        def __init__(self, embedding_dim):
                super().__init__()
                self.embedding_dim = embedding_dim

        def forward(self, z_e, embed_weight):
                flatten = z_e.reshape(-1, self.embedding_dim)
                dist = (
                        flatten.pow(2).sum(1, keepdim=True)
                        - 2 * flatten @ embed_weight.t()
                        + embed_weight.pow(2).sum(1, keepdim=True).t()
                )
                _, embed_ind = (-dist).max(1)
                embed_ind = embed_ind.view(*z_e.shape[:-1])
                z_q = self.embed_code(embed_ind, embed_weight)

                commitment_cost = 0.25
                diff = commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()

                z_q = z_e + (z_q - z_e).detach()

                return z_q, diff, embed_ind

        def embed_code(self, embed_id, embed_weight):
                return F.embedding(embed_id, embed_weight)