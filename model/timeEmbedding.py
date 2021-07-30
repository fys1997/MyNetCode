
import torch.nn as nn


class timeEmbedding(nn.Module):
    def __init__(self, num_embedding, embedding_dim):
        super(timeEmbedding, self).__init__()
        self.embed = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim)

    def forward(self,x):
        """

        :param x: 时间：[batch*T]
        :return:batch*T*embedding_dim
        """
        x=x.long()
        return self.embed(x)
