import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEmbedding(nn.Module):
    def __init__(self, max_position_embeddings: int, embedding_size: int) -> None:
        super().__init__()
        self.pos_emb = nn.Embedding(max_position_embeddings, embedding_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(-1)
        device = input_ids.device
        positions = torch.arange(seq_len, device=device)
        return self.pos_emb(positions)

class EmbeddingFactorized(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            max_position_embeddings: int,
            hidden_size: int = 64,
            embedding_size: int = 256,
            pad_token_id: int = 0
        ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.linear = nn.Linear(hidden_size, embedding_size)
        self.pos_embedding = PositionalEmbedding(max_position_embeddings, embedding_size)
        self.segment_embeddings = nn.Embedding(2, embedding_size)

    def forward(self, token_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.linear(self.embedding(token_ids))
        pos_embeddings = self.pos_embedding(token_ids)
        segment_embeddings = self.segment_embeddings(token_type_ids)

        return embeddings + pos_embeddings + segment_embeddings

class Albert(nn.Module):
    def __init__(
            self,
            vocab_size: int = 30000,
            max_position_embeddings: int = 1024,
            hidden_size: int = 64,
            embedding_size: int = 256,
            num_layers: int = 8,
            num_attention_heads: int = 8,
            intermediate_size: int = 768,
            dropout: float = 0.1,
            pad_token_id: int = 0,
        ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.num_layers = num_layers

        self.embedding = EmbeddingFactorized(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            pad_token_id=pad_token_id
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            batch_first=True
        )

        self.mlm_head = nn.Linear(embedding_size, vocab_size)
        self.sop_head = nn.Linear(embedding_size, 2)
    
    def forward(
            self,
            input_ids: torch.Tensor,
            token_type_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            mlm_labels: torch.Tensor | None = None,
            sop_label: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, dict[str, torch.Tensor]]:

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embeddings = self.embedding(input_ids, token_type_ids)
        

        if attention_mask is not None:
            padding_mask = attention_mask.eq(0)
        else:
            padding_mask = input_ids.eq(self.pad_token_id)


        for _ in range(self.num_layers):
            embeddings = self.encoder_layer(embeddings, src_key_padding_mask=padding_mask)


        mlm_logits = self.mlm_head(embeddings)
        pooled = embeddings[:, 0, :]
        sop_logits = self.sop_head(pooled)

        if mlm_labels is None and sop_label is None:
            return {"mlm_logits": mlm_logits, "sop_logits": sop_logits}

        loss = None
        mlm_loss = None
        sop_loss = None
        if mlm_labels is not None:
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                mlm_labels.view(-1),
                ignore_index=-100,
            )
            loss = mlm_loss if loss is None else loss + mlm_loss

        if sop_label is not None:
            sop_loss = F.cross_entropy(sop_logits, sop_label)
            loss = sop_loss if loss is None else loss + sop_loss

        return loss, mlm_loss, sop_loss, {"mlm_logits": mlm_logits, "sop_logits": sop_logits}