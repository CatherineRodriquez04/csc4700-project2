import sys
import os
import torch
from torch import nn
import torch.nn.functional as F

## get the parent directory and add to sys.path
current_path = os.path.dirname(__file__)
root_path = os.path.abspath(os.path.join(current_path, ".."))
sys.path.append(root_path)

from base_model import Base_Model

class Transformer_Model(Base_Model):
    def __init__(self, vocab_size, embedding_dim=256, num_layers=6, nhead=2, max_seq_len=512, dropout=0.2, padding_id=3):
        '''
        Transformer-based language model for sequence generation.

        Arguments:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimensionality of token embeddings.
            num_layers (int): Number of transformer encoder layers.
            nhead (int): Number of attention heads per layer.
            max_seq_len (int): Maximum sequence length supported.
            dropout (float): Dropout probability.
            padding_id (int): ID of the padding token.
        '''
        # initialize base class with embedding setup
        super().__init__(vocab_size, embedding_dim, padding_id, model_name="Transformer")

        # positional embeddings to encode token positions in the sequence
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # define a single encoder layer and stack them to build the full encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # final linear layer to map transformer outputs to vocabulary logits
        self.fc = nn.Linear(embedding_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tokens):
        '''
        Performs a forward pass through the Transformer model.

        Arguments:
            input_tokens (Tensor): Input token IDs 

        Return:
            Tuple[Tensor, None]: 
                - Logits of shape (batch_size, sequence_length, vocab_size)
                - None (placeholder for consistency with RNN models)
        '''
        batch_size, seq_len = input_tokens.size()
        token_embeddings = self.embedding(input_tokens)

        # generate and expand positional indices
        positions = torch.arange(seq_len, device=input_tokens.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embeddings = self.pos_embedding(positions)

        embeddings = self.dropout(token_embeddings + pos_embeddings)

        # create mask 
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_tokens.device) * float('-inf'), diagonal=1)

        # pass through the transformer encoder
        transformer_output = self.transformer_encoder(embeddings, mask=mask)

        logits = self.fc(transformer_output)
        return logits, None
