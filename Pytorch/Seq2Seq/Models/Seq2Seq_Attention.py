import torch
import torch.nn as nn
import random
import sys

class Encoder(nn.Module):
    '''Encode 1 sentence at a time'''
    def __init__(self, source_vocab_size, embedding_size, hidden_size, num_layers=1, dropout_rate=0.4):
        super(Encoder, self).__init__()

        self.source_vocab_size = source_vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = 1

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.source_vocab_size,
            embedding_dim=self.embedding_size)
        self.dropout = nn.Dropout(dropout_rate)

        # RNN: biLSTM layer
        self.rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            dropout=dropout_rate)

        # Fully connected layer
        self.cat_hidden = nn.Linear(
            in_features=2*self.hidden_size,
            out_features=self.hidden_size)
        self.cat_cell = nn.Linear(
            in_features=2*self.hidden_size,
            out_features=self.hidden_size)

    def forward(self, source):
        # source: (source_input_size, batchsize)

        ## Embedding layer
        embedded = self.embedding(source)
        embedded_dr = self.dropout(embedded)
            # embedded: (source_input_size, batchsize, embedding_size)

        ## RNN: 1 biLSTM layer
        encoder_states, (hidden, cell) = self.rnn(embedded_dr)
            # encoder_states: (source_input_size, batchsize, 2*hidden_size)
            # hidden: (2, batchsize, hidden_size)
            # cell: (2, batchsize, hidden_size)

        # Notes: num_layer = 1 -> we concat foward and backward h,c into 1
        #   hidden/cell[0]: bi-LSTM forward
        #   hidden/cell[1]: bi-LSTM backward
        hidden = self.cat_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.cat_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))
            # hidden: (1, batchsize, hidden_size)
            # cell: (1, batchsize, hidden_size)

        return encoder_states, hidden, cell


class Decoder(nn.Module):
    '''Doing prediction 1 word at a time'''
    def __init__(self, target_vocab_size, embedding_size, hidden_size, num_layers=1, dropout_rate=0.4):
        super(Decoder, self).__init__()

        self.target_vocab_size = target_vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.num_layers = 1

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.target_vocab_size,
            embedding_dim=self.embedding_size)
        self.dropout = nn.Dropout(dropout_rate)

        # Attention
        self.energy_fc = nn.Linear(
            in_features=3*hidden_size,
            out_features=1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

        # RNN: LSTM layer
        self.rnn = nn.LSTM(
            input_size=2*self.hidden_size + self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers)

        # Fully Connected layer
        self.fc = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.target_vocab_size)
        

    def forward(self, X, encoder_states, hidden, cell):
        # X: (batchsize)
        # encoder_states: (source_input_size, batchsize, 2*hidden_size)
        # hidden: (1, batchsize, hidden_size)
        # cell: (1, batchsize, hidden_size)

        ## Embedding layer
        X_unsq = X.unsqueeze(0)
            # X_unsq: (1, batchsize)
        embedded = self.embedding(X_unsq)
        embedded_dr = self.dropout(embedded)
            # embedded: (1, batchsize, embedding_size)

        ## Attention - Calc context_vector
        source_input_size = encoder_states.shape[0]
        h_reshaped = hidden.repeat(source_input_size, 1, 1)
            # h_reshaped: (source_input_size, batch_size, hidden_size)
        inp_cat = torch.cat((h_reshaped, encoder_states), dim=2)
            # inp_cat: (source_input_size, batch_size, 3*hidden_size)

        energy = self.energy_fc(inp_cat)
        energy_relu = self.relu(energy)
        alpha = self.softmax(energy_relu)
            # alpha: (source_input_size, batch_size, 1)
        context_vector = torch.einsum("snk,snl->knl", alpha, encoder_states)
            # context_vector: (1, batch_size, 2*hidden_size)

        ## RNN: LSTM layer
        rnn_input = torch.cat((context_vector, embedded_dr), dim=2)
            # rnn_input: (1, batch_size, 2*hidden_size + embedding_size)
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
            # outputs: (1, batchsize, hidden_size)
            # hidden: (1, batchsize, hidden_size)
            # cell: (1, batchsize, hidden_size)

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
        encoder_states, hidden, cell = self.encoder(source)
            # hidden: (num_layers, batchsize, hidden_size)
            # cell: (num_layers, batchsize, hidden_size)

        # Grab the first word input to the Decoder which will be <SOS> token
        X = target[0]
            # X: (batchsize)

        # Predict the target sentence word by word
        for t in range(1, target_input_size):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(X, encoder_states, hidden, cell)
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