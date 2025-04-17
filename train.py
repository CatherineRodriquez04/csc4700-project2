import torch
import sentencepiece as spm
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
from pathlib import Path
from language_dataset import LanguageDataset, prepare_batch

BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 3  


def plot_loss_curve(train_losses, val_losses, model_name):
    '''
        Once trained, the model with be plotted on a graph to display loss curve
    '''
    os.makedirs("model_graphs", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve for {model_name}")
    plt.legend()
    plt.grid()
    plt.savefig(f"model_graphs/{model_name}.png")
    plt.close()

def train_model(model, model_name):
    '''
    Trains a language model and saves the best-performing version based on validation loss.

    Args:
        model (nn.Module): The initialized model to train
        model_name (str): Name of the model architecture ("RNN", "LSTM", or "Transformer")

    Returns:
        None. Saves the best model to disk and plots the training/validation loss curves.
    '''
    # ensure training and validation data exist
    assert Path("data/train_processed.jsonl").exists(), "The training dataset cannot be found"
    assert Path("data/test_processed.jsonl").exists(), "The validation dataset cannot be found"

    # Prepare datasets and dataloaders
    train_dataset = LanguageDataset("data/train_processed.jsonl", tokenizer, max_len=128)
    val_dataset = LanguageDataset("data/test_processed.jsonl", tokenizer, max_len=128)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=prepare_batch)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=prepare_batch)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=3)  # ignore padding token in loss
    best_val_loss = float('inf')
    no_improve_epochs = 0
    train_losses, val_losses = [], []

    # start training loop
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        # training step
        for input_ids, target_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            optimizer.zero_grad()
            logits, _ = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # calculate and store average training loss
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # validation step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids, target_ids = input_ids.to(device), target_ids.to(device)
                logits, _ = model(input_ids)
                loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
                total_val_loss += loss.item()

        # calculate and store average validation loss
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")

        # update learning rate scheduler
        scheduler.step(avg_val_loss)

        # save model if it has improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = f"{model_name}/trained_{model_name}.pth"
            os.makedirs(model_name, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved at {checkpoint_path}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"No improvement. Patience left: {PATIENCE - no_improve_epochs}")

        # early stopping check
        if no_improve_epochs >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    # plot training and validation loss curves
    plot_loss_curve(train_losses, val_losses, model_name)


if __name__ == "__main__":
    # parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a neural language model.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["rnn", "lstm", "transformer"],
        required=True,
        help="Choose the model architecture to train"
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    # load tokenizer and vocabulary
    tokenizer = spm.SentencePieceProcessor(model_file='tokenizer/bpe_tokenizer.model')
    vocab_size = tokenizer.get_piece_size()

    # map model name to class
    model_classes = {
        "rnn": "RNN_Model",
        "lstm": "LSTM_Model",
        "transformer": "Transformer_Model"
    }

    # import the selected model module and class
    module = __import__(args.model, fromlist=[model_classes[args.model]])
    ModelClass = getattr(module, model_classes[args.model])
    model = ModelClass(vocab_size=vocab_size).to(device)

    # train the model
    train_model(model, args.model)
