import torch
from torch.utils.data import Dataset
import json
from torch import nn

def prepare_batch(batch):
    """
    Collate function to pad input and target sequences for uniform batch size.

    Arguments:
        batch (list of tuples): Each tuple contains (input_ids, target_ids) of varying lengths

    Return:
        Tuple[Tensor, Tensor]: Padded input and target tensors of shape (batch_size, seq_len)
    """
    input_seqs, target_seqs = zip(*batch)

    # pad input and sequences to the same length
    input_padded = nn.utils.rnn.pad_sequence(input_seqs, batch_first=True, padding_value=3)
    target_padded = nn.utils.rnn.pad_sequence(target_seqs, batch_first=True, padding_value=3)

    return input_padded, target_padded


class LanguageDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        """
        Dataset class for loading and preparing data for causal language modeling.

        Arguments:
            data_path (str): Path to the JSONL dataset file
            tokenizer: SentencePiece tokenizer for encoding text
            max_len (int): Maximum allowed token length per sample
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        # Load and process each line in the dataset
        with open(data_path, "r", encoding="utf-8") as file:
            for line in file:
                item = json.loads(line.strip())
                # Combine prompt and completion into a single string
                prompt = item.get("prompt", "")
                completion = item.get("completion", "")
                text = f"{prompt} {completion}".strip()

                # Tokenize and truncate to max_length
                token_ids = tokenizer.encode(text, out_type=int)[:max_len]

                # Discard very short sequences
                if len(token_ids) < 2:
                    continue

                self.samples.append(token_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Retrieves a tokenized sample and splits it into input and target sequences.

        Arguments:
            index (int): Index of the sample to retrieve

        Return:
            Tuple[Tensor, Tensor]: input_ids (all tokens except last), 
                                   target_ids (all tokens except first)
        """
        tokens = self.samples[index]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids