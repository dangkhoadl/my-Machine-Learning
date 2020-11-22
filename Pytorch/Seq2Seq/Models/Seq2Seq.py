import torch
import torch.nn as nn
import random
import sys

class Encoder(nn.Module):
    '''Encode 1 sentence at a time'''
    def __init__(self, source_vocab_size, embedding_size, hidden_size, num_layers, dropout_rate=0.4):
        super(Encoder, self).__init__()

        self.source_vocab_size = source_vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.source_vocab_size,
            embedding_dim=self.embedding_size)
        self.dropout = nn.Dropout(dropout_rate)

        # RNN: LSTM layer
        self.rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropout_rate)

    def forward(self, source):
        # source: (source_input_size, batchsize)

        ## Embedding layer
        embedded = self.embedding(source)
        embedded_dr = self.dropout(embedded)
            # embedded: (source_input_size, batchsize, embedding_size)

        ## RNN: LSTM layer
        outputs, (hidden, cell) = self.rnn(embedded_dr)
            # outputs: (source_input_size, batchsize, hidden_size)
            # hidden: (num_layers, batchsize, hidden_size)
            # cell: (num_layers, batchsize, hidden_size)

        return hidden, cell


class Decoder(nn.Module):
    '''Doing prediction 1 word at a time'''
    def __init__(self, target_vocab_size, embedding_size, hidden_size, num_layers, dropout_rate=0.4):
        super(Decoder, self).__init__()

        self.target_vocab_size = target_vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.num_layers = num_layers

        ## Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.target_vocab_size,
            embedding_dim=self.embedding_size)
        self.dropout = nn.Dropout(dropout_rate)

        ## RNN: LSTM layer
        self.rnn = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropout_rate)

        ## Fully Connected layer
        self.fc = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.target_vocab_size)

    def forward(self, X, hidden, cell):
        # X: (batchsize)
        # hidden: (num_layers, batchsize, hidden_size)
        # cell: (num_layers, batchsize, hidden_size)


        ## Embedding layer
        X_unsq = X.unsqueeze(0)
            # X_unsq: (1, batchsize)
        embedded = self.embedding(X_unsq)
        embedded_dr = self.dropout(embedded)
            # embedded: (1, batchsize, embedding_size)

        ## RNN: LSTM layer
        outputs, (hidden, cell) = self.rnn(embedded_dr, (hidden, cell))
            # outputs: (1, batchsize, hidden_size)
            # hidden: (num_layers, batchsize, hidden_size)
            # cell: (num_layers, batchsize, hidden_size)

        ## Fully Connected layer
        predictions = self.fc(outputs)
            # predictions: (1, batchsize, target_vocab_size)
        predictions_squ = predictions.squeeze(0)
            # predictions_squ: (batchsize, target_vocab_size)

        return predictions_squ, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, target_vocab_size, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.device = device
        self.target_vocab_size = target_vocab_size

    def forward(self, source, target, teacher_force_ratio=0.5):
        # source: (source_input_size, batch_size)
        # target: (target_input_size, batch_size)

        batch_size = source.shape[1]
        target_input_size = target.shape[0]

        # Initialize output tensor
        outputs = torch.zeros(target_input_size, batch_size, self.target_vocab_size).to(self.device)
            # outputs: (target_input_size, batch_size, target_vocab_size)

        # Encode source sentence
        hidden, cell = self.encoder(source)
            # hidden: (num_layers, batchsize, hidden_size)
            # cell: (num_layers, batchsize, hidden_size)

        # Grab the first word input to the Decoder which will be <SOS> token
        X = target[0]
            # X: (batchsize)

        # Predict the target sentence word by word
        for t in range(1, target_input_size):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(X, hidden, cell)
                # output: (batchsize, target_vocab_size)
                # hidden: (num_layers, batchsize, hidden_size)
                # cell: (num_layers, batchsize, hidden_size)

            # Store next output prediction
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)
                # best_guess: (batchsize, target_vocab_size)

            # Select the next target word based on teacher_force_ratio
            #   50%: correct one from target
            #   50%: the prediction from decoder
            X = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
