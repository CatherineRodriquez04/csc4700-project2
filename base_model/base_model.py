import torch
from torch import nn
import torch.nn.functional as F

class Base_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_id, model_name):
        """
        This is the base class for language models.

        Arguments:
            vocab_size: Number of unique tokens in the vocabulary
            embedding_dim: Dimensionality of the embedding space
            padding_id: Index of the padding token
            model_name: Model name used for training
        """
        super(Base_Model, self).__init__()
        self.name = model_name
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_id)


    def predict(self, input_ids, temp=1.0, bos_token_id=None):
        """
        Predict the next token ID (and hidden state) from the last token in input_ids.

        Arguments:
            input_ids: Input sequence token IDs
            temp (float): Temperature for sampling randomness

        Return:
            Tuple[int, Any]: Predicted token index, and optional hidden state
        """
        self.eval() # set the model to evaluation mode which disables dropout
        with torch.no_grad():
            logits, hidden = self.forward(input_ids)

            # use the last token's logits and apply temperature
            logits = logits[:, -1, :] / (temp if temp > 0 else 1.0)
            
            # block the <bos> token from being chosen again
            if bos_token_id is not None:
                logits[:, bos_token_id] = -float('inf')  
            
            # convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.argmax(probs, dim=-1) if temp == 0 else torch.multinomial(probs, num_samples=1).squeeze(-1)

            return next_token_id.item(), hidden
        

    def generate(self, tokenizer, prompt, max_len=50, temp=1.0, device='cpu'):
        """
        Generate text sequence of tokens from a given prompt.

        Arguments:
            tokenizer: Tokenizer utility with encode/decode methods
            prompt (str): Starting string to guide generation
            max_len (int): Maximum number of tokens to produce
            temp (float): Temperature for sampling randomness
            device (str): Device to run inference on

        Return: 
            str: Fully generated string
        """
        eos_token_id = tokenizer.eos_id()

        self.eval() # set the model to evaluation mode which disables dropout

        # Convert the input prompt into token IDs and move it to the correct device
        id_input = tokenizer.encode(prompt, out_type=int)
        tensor_input = torch.tensor(id_input, dtype=torch.long, device=device).unsqueeze(0)

        generated_ids = []
        hidden = None

        for _ in range(max_len):
            # Predict the next token using the model
            next_token_id, hidden = self.predict(
                tensor_input, temp, bos_token_id=tokenizer.bos_id()
            )
            
            # Stop generating if <eos> is reached
            if eos_token_id is not None and next_token_id == eos_token_id:
                break

            # Add the predicted token to the list
            generated_ids.append(next_token_id)

            # Update the input tensor with the newly predicted token
            tensor_input = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

        return tokenizer.decode(generated_ids, out_type=str)