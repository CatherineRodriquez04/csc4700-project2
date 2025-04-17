import torch
import sentencepiece as spm
import argparse
import os

if __name__ == "__main__":
    # argument parser for selecting which model to generate text
    parser = argparse.ArgumentParser(description="Run text generation using a pretrained model.")
    parser.add_argument("prompt", type=str, help="Starting text to condition the generation process")
    parser.add_argument("--model", type=str, default="rnn", help="Choose a model: rnn, lstm, or transformer")
    parser.add_argument("--temp", type=float, default=1.0, help="Sampling temperature (higher = more random)")
    parser.add_argument("--max_len", type=int, default=50, help="Maximum number of tokens to generate")
    args = parser.parse_args()
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else 
        "cpu"
    )


    tokenizer = spm.SentencePieceProcessor("tokenizer/bpe_tokenizer.model")
    vocab_size = tokenizer.get_piece_size()

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


    generated_text = model.generate(tokenizer, args.prompt, temp=args.temp, device=device, max_len=args.max_len)

    # print results
    print("Generated Text:")
    print(generated_text)

    