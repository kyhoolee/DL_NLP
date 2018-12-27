# Neural Network Language Model
## Introduction
- A general package to learn neural network language models. They are:

  + RNN (BiRNN), deep-RNN (deep-biRNN)
  + GRU (BiGRU), deep-GRU (deep-biGRU)
  + LSTM (BiLSTM), deep-LSTM (deep-biLSTM)
  
- Inputs:
  + Train_data: used for training a model
  + Dev_data: used for validating a model
  + Test_data: used for testing a model
- Outputs:
  + A trained-model file
  + A model-argument file
- Applications:
  + Text generation
    - P(qj)=P(w1,w2,...,wn)
  + Text recommendation
    - P(wi|wi-1,...,w1)

## Folder Structure
  1. model.py: main script to train the model
  2. interactive.py: an interface to generate a query
  3. predict.py: a function to predict if two queries are concatenation or separation
  4. generate.py: a funtion to generate a list of queries avd save into a file
  5. utils/
    - core_nns.py: a script to design the neural network architecture
    - data_utils.py: a script to load data and convert to indices
  6. results/: a folder to save the train model
    - *.m: a trained-model file
    - *.args: a model-argument file
## Package workflow
1. Trained mode
- Design a model architecture (mainly done in utils/core_nns.py)
- Load training and testing data map to indices (mainly done in utils/data_utils.py)
- Feed data to model to train it (mainly done in model.py)
- Save model into files (mainly done in model.py)
- Stop the training process (mainly done in model.py) when  
    + having no improvement on testing data or;
    + exceeding the maximum number of epochs
2. Predicted mode
- Load the save files at training phase
- Write ``predict.py`` file to predict the next words given some initial words: P(wi|wi-1,...,w1) and generate a piece of text.

## Project usage
1. Download this repository: git clone https://github.com/duytinvo/DL_NLP
2. Train the model:
  - Change the current directory to "nnlm"
  - Run this command:
```
python model.py --train_file /media/data/queries/training_set_180712_rev00.csv --test_file /media/data/queries/test_set_180712_rev00.csv --trained_model ./results/model.query.3l.fr180712.pt --model_args ./results/model.query.3l.fr180712.args
```
  - See other parameters, this command:
```
python model.py -h
```

