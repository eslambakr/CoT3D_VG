import torch
import torch.nn as nn


class CoTTransformer(nn.Module):
    def    __init__(
        self,
        embedding_size,
        num_cls,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        trg_len,
    ):
        super(CoTTransformer, self).__init__()
        self.src_position_embedding = nn.Embedding(52, embedding_size)
        self.trg_position_embedding = nn.Embedding(trg_len, embedding_size)
        self.transformer = nn.Transformer(embedding_size, num_heads, num_encoder_layers, num_decoder_layers,
                                          forward_expansion, dropout)
        self.fc_out = nn.Linear(embedding_size, num_cls)
        self.dropout = nn.Dropout(dropout)
        self.device = None

    def forward(self, embed_src, query_embed, device):
        """
        embed_src: [src_seq_length, N, embed]
        embed_trg: [trg_seq_length, N, embed]
        """
        self.device = device
        src_seq_length, N, _ = embed_src.shape
        trg_seq_length, _ = query_embed.shape

        # [trg_seq_length, N]
        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )

        # [src_seq_length, N, embed]
        embed_src = self.dropout(embed_src)  # There is no order for the input sequence.

        # Create masks:
        # [trg_seq_length, trg_seq_length]
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)
        #trg_mask = None
        src_mask = torch.zeros((src_seq_length, src_seq_length), device=self.device).type(torch.bool)
        #src_mask = None

        # Create Target embedding:
        embed_trg = query_embed.unsqueeze(1).repeat(1, N, 1)  # [trg_seq_length, N, embed]
        embed_trg = self.dropout(embed_trg + self.trg_position_embedding(trg_positions))

        out = self.transformer(embed_src, embed_trg, tgt_mask=trg_mask, src_mask=src_mask)  # [trg_seq_length, N, embed]
        out = out.permute(1, 0, 2)  # [trg_seq_length, N, embed] --> [N, trg_seq_length, embed]
        out = self.fc_out(out)  # [N, trg_seq_length, embed] --> [N, trg_seq_length, num_cls]
        return out
