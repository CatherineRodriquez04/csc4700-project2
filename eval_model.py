import torch
import sentencepiece as spm
import numpy as np
from torch.utils.data import DataLoader
from language_dataset import LanguageDataset
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import argparse
import os
from language_dataset import prepare_batch

def calc_perplexity(model, data_loader, criterion, vocab_size, device):
    '''
    Calculates the perplexity of a language model on a given dataset.

    Arguments:
        model (nn.Module): The trained language model.
        data_loader (DataLoader): Dataloader providing input-target pairs
        criterion (nn.Module): Loss function (typically CrossEntropyLoss)
        vocab_size (int): Size of the vocabulary (used to reshape logits)
        device (torch.device): Device to perform computation on (CPU or GPU)

    Return:
        float: The computed perplexity score (lower is better)
    '''
    # set model to evaluation mode
    model.eval()
    total_loss = 0
    total_tokens = 0

    # disable gradient
    with torch.no_grad():
        for input_ids, target_ids in tqdm(data_loader, desc="Computing Perplexity"):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            logits, _ = model(input_ids)

            # compute the loss (cross-entropy between predictions and targets)
            # flatten both logits and targets for loss computation
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))

            # calculate loss
            total_loss += loss.item() * target_ids.numel()
            total_tokens += target_ids.numel()

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity


def calc_bleu_score(model, data_loader, tokenizer, device):
    '''
    Computes the BLEU score of a language model's predictions against target sequences.

    Arguments:
        model (nn.Module): The trained language model
        data_loader (DataLoader): Dataloader providing input-target pairs
        tokenizer: Tokenizer with decode method to convert token IDs to strings
        device (torch.device): Device to perform computation on (CPU or GPU)

    Return:
        float: The corpus-level BLEU score (higher is better)
    '''
    # set model to evaluation mode 
    model.eval()

    ref = []
    hypo = []

    # disable gradient
    with torch.no_grad():
        for input_ids, target_ids in tqdm(data_loader, desc="Computing BLEU Score"):
            # move input tensors to the correct device
            input_ids = input_ids.to(device)

            # generate predictions from the model
            logits, _ = model(input_ids)

            # get likely token at each position
            predicted_ids = torch.argmax(logits, dim=-1).cpu().tolist()
            target_ids = target_ids.cpu().tolist()

            # decode token IDs to text and store references/hypotheses
            for pred, target in zip(predicted_ids, target_ids):
                pred_text = tokenizer.decode(pred)
                target_text = tokenizer.decode(target)

                ref.append([target_text.split()])
                hypo.append(pred_text.split())

    # compute corpus-level BLEU score
    bleu_score = corpus_bleu(ref, hypo)

    return bleu_score

if __name__ == "__main__":
    # argument parser for selecting which model to evaluate
    parser = argparse.ArgumentParser(description="Evaluate a trained model using Perplexity and BLEU Score")
    parser.add_argument("--model", type=str, default="rnn", help="Choose a model: rnn, lstm, transformer")
    args = parser.parse_args()
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else 
        "cpu"
    )
    
    # load tokenized vocab & get vocab size
    tokenizer = spm.SentencePieceProcessor("tokenizer/bpe_tokenizer.model")
    vocab_size = tokenizer.get_piece_size()
    
    test_dataset = LanguageDataset("data/test_processed.jsonl", tokenizer, 128)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=prepare_batch)
    
    # load the specified model architecture and its trained weights
    if args.model == "rnn":
        from rnn import RNN_Model
        model = RNN_Model(vocab_size=vocab_size).to(device)
        checkpoint = "trained_rnn.pth"

    elif args.model == "lstm":
        from lstm import LSTM_Model
        model = LSTM_Model(vocab_size=vocab_size).to(device)
        checkpoint = "trained_lstm.pth"

    elif args.model == "transformer":
        from transformer import Transformer_Model
        model = Transformer_Model(vocab_size=vocab_size).to(device)
        checkpoint = "trained_transformer.pth"

    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    
    # load the saved model state from the appropriate file
    model.load_state_dict(torch.load(os.path.join(args.model, checkpoint), map_location=device))

    # define the loss function (ignores padding token with ID 3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=3) 
    
    # calculate scores
    ppl = calc_perplexity(model, test_loader, criterion, vocab_size, device)
    bleu = calc_bleu_score(model, test_loader, tokenizer, device)
    
    # print results
    print(f"Perplexity: {ppl:.4f}")
    print(f"BLEU Score: {bleu:.4f}")