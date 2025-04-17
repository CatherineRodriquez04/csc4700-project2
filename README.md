# Neural Language Model
This project implements and compares three different language modeling architectures: `RNN`, `LSTM`, and `Transformer`. It supports training, evaluation, and text generation on custom datasets using a SentencePiece tokenizer.

## Tokenization Setup
The data used for this project is located within the data folder that is located in the root, e.g. `csc4700-project2/data` .

If you ever need to retrain the Tokenizer, start by moving into the Tokenizer directory with cd tokenizer, then execute `python3 tokenizer.py`. This process not only retrains the model but also prepares the dataset by inserting start (<bos>) and end (<eos>) tokens into the training and testing files, and produces a corpus.txt file for reference. Note that the Tokenizer has already been trained in this repository.

## Training the Models
The model is already trained, but you can retrain it if you'd like. Follow the steps below:

1. Ensure your dataset is tokenized. 
   If your data hasn't been tokenized yet, refer to the **Tokenization Setup** section before proceeding.

2. Navigate to the root directory of the project.

3. Run the training script with your desired model architecture:

- `python3 train.py --model [model]`

- Replace [model] with one of the following:
   - rnn
   - lstm
  - transformer

4. Trained weights will be saved in a folder named after the model, e.g., `rnn/trained_rnn.pth`.

## Text Generation
Once you've trained or downloaded a model, you can generate text from a custom prompt using the 'generate_text.py' script.

Use the following command in the root folder, e.g. `csc4700-project2'`:

- `python3 generate_text.py --model [model] --max_len [max_length] --temp [temperature]`

- Replace [model] with one of the following:
  - rnn
  - lstm
  - transformer
- [max_length] can be an int 0 or higher
- [temperature] can be a float 0.0 (greedy) or higher

## Evaluatation
To evaluate the performance of your trained language model, two key metrics were used:

1. Perplexity – to measure how well the model predicts text.
2. BLEU Score – to assess the quality of the generated text against reference outputs.

### Perplexity
Perplexity measures how confident the model is in its predictions. It’s based on the model’s cross-entropy loss.
- Lower perplexity means better model performance (more accurate predictions)

### Bleu Score
The BLEU score evaluates the quality of generated text by comparing it to reference (target) text using n-gram overlaps.
- Higher BLEU scores reflect more fluent and accurate text generation.
- A BLEU score close to 1.0 indicates strong alignment between generated and reference outputs.

### How to Compute Perplexity and BLEU Scores
Use the following command in the root folder, e.g. 'csc4700-project2'

`python3 eval_model.py --model [model]`

Replace [model] with one of the following:
- rnn
- lstm
- transformer

