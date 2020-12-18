import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embedding_size, heads):
        super(SelfAttention, self).__init__()

        self.embedding_size = embedding_size
        self.heads = heads
        self.head_dim = embedding_size // heads

        assert (
            self.head_dim * heads == embedding_size
        ), "Embedding size needs to be divisible by heads"

        # Linears
        self.V_linear = nn.Linear(
            in_features=self.head_dim,
            out_features=self.head_dim, bias=False)
        self.K_linear = nn.Linear(
            in_features=self.head_dim,
            out_features=self.head_dim, bias=False)
        self.Q_linear = nn.Linear(
            in_features=self.head_dim,
            out_features=self.head_dim, bias=False)

        # Out
        self.fc_out = nn.Linear(
            in_features=heads * self.head_dim,
            out_features=embedding_size)

    def forward(self, values, keys, query, mask):
        batch_size = query.size(0)
        value_len, key_len, query_len = values.size(1), keys.size(1), query.size(1)

        # split embedding into head pieces
        values = values.reshape(batch_size, value_len, self.heads, self.head_dim)
        keys = keys.reshape(batch_size, key_len, self.heads, self.head_dim)
        query = query.reshape(batch_size, query_len, self.heads, self.head_dim)

        values_ln = self.V_linear(values)
        keys_ln = self.K_linear(keys)
        queries_ln = self.Q_linear(query)


        energy = torch.einsum("nqhd,nkhd->nhqk", [queries_ln, keys_ln])


        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embedding_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values_ln]).reshape(
            batch_size, query_len, self.heads * self.head_dim
        )

        out_fc = self.fc_out(out)

        return out_fc


class EncoderBlock(nn.Module):
    def __init__(self, embedding_size, heads, dropout_rate, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.attention = SelfAttention(
            embedding_size=embedding_size,
            heads=heads)
        self.norm1 = nn.LayerNorm(normalized_shape=embedding_size)
        self.norm2 = nn.LayerNorm(normalized_shape=embedding_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, forward_expansion * embedding_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embedding_size, embedding_size),
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, value, key, query, mask):
        # Multi-Head Attention
        attention = self.attention(
            value=value, key=key, query=query,
            mask=mask)

        # Add & Norm
        norm_1 = self.norm1(attention + query)
        x = self.dropout(norm_1)

        # Feed forward
        forward = self.feed_forward(x)

        # Add & Norm
        norm_2 = self.norm2(forward + x)
        out = self.dropout(norm_2)
        return out


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embedding_size, num_layers,
            heads, device, forward_expansion, dropout_rate, max_length):
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.device = device

        self.word_embedding = nn.Embedding(
            num_embeddings=src_vocab_size,
            embedding_dim=embedding_size)
        self.position_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embedding_size)
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embedding_size=embedding_size,
                    heads=heads,
                    dropout=dropout_rate,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        batch_size, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(batch_size, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out