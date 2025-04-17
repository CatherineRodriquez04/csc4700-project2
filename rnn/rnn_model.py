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

class RNN_Model(Base_Model):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=4, dropout=0.2, padding_id=3):
        """
        RNN-based neural model for generating language sequences.

        Arguments:
            vocab_size (int): Total number of tokens in the vocabulary
            embedding_size (int): Dimension of token embeddings
            hidden_size (int): Size of hidden layers in the RNN
            layers (int): Number of stacked RNN layers
            dropout_rate (float): Dropout between RNN layers for regularization
            padding_id (int): Token ID used for padding (<pad> token)
        """
        # initalize the base class with the model name
        super().__init__(vocab_size, embedding_dim, padding_id, model_name="RNN")

        # define RNN module with stacked layers and dropout
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, nonlinearity='tanh')

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None):
        '''
        Runs a forward pass through the RNN model.

        Arguments:
            input_ids (Tensor): Input token IDs with shape (batch_size, sequence_length)
            hidden (Tensor, optional): Initial hidden state for the RNN 

        Return:
            Tuple[Tensor, Tensor]:
                - Logits over vocabulary for each token position (batch_size, sequence_length, vocab_size)
                - Final hidden state from the RNN
        '''
        # convert input token IDs to embeddings
        token_embeddings = self.embedding(input_ids)

        # pass embeddings through the RNN
        rnn_output, hidden = self.rnn(token_embeddings, hidden)

        # apply the output projection to get logits
        logits = self.fc(rnn_output)

        return logits, hidden
