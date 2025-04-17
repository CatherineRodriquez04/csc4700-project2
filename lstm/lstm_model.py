import sys
import os
import torch
from torch import nn
import torch.nn.functional as F

# get the parent directory and add to sys.path
current_path = os.path.dirname(__file__)
root_path = os.path.abspath(os.path.join(current_path, ".."))
sys.path.append(root_path)

from base_model import Base_Model

class LSTM_Model(Base_Model):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=5, dropout=0.2, padding_id=3):
        """
        LSTM-based neural model for generating language sequences.

        Arguments:
            vocab_size (int): Total number of tokens in the vocabulary
            embedding_size (int): Dimension of token embeddings
            hidden_size (int): Size of hidden layers in the LSTM
            layers (int): Number of stacked LSTM layers
            dropout_rate (float): Dropout between LSTM layers for regularization
            padding_id (int): Token ID used for padding (<pad> token)
        """
        # initalize the base class with the model name
        super().__init__(vocab_size, embedding_dim, padding_id, model_name="LSTM")

        # define LSTM module with stacked layers and dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_tokens, hidden=None):
        """
        Forward pass for the LSTM model to compute prediction scores.

        Arguments:
            input_tokens (Tensor): Input tensor of shape (batch_size, sequence_length)

        Return:
            Tuple[Tensor, tuple]: 
                - Logits of shape (batch_size, sequence_length, vocab_size)
                - Final LSTM state (hidden and cell)
        """
        embeds = self.embedding(input_tokens)
        output, hidden = self.lstm(embeds, hidden)
        return self.fc(output), hidden
