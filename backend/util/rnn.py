import os

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import numpy as np


from util.helper import seg_char, pad_input, one_hot_encode


TRAIN_FILE = "util/rnn_train_dataset.txt"
MODEL_FILE = "util/rnn.model"

KHCONST = list(u"កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអឣឤឥឦឧឨឩឪឫឬឭឮឯឰឱឲឳ")
KHVOWEL = list(u"឴឵ាិីឹឺុូួើឿៀេែៃោៅ\u17c6\u17c7\u17c8")
# subscript, diacritics
KHSUB = list(u"្")
KHDIAC = list(
    u"\u17c9\u17ca\u17cb\u17cc\u17cd\u17ce\u17cf\u17d0"
)  # MUUSIKATOAN, TRIISAP, BANTOC,ROBAT,
KHSYM = list("៕។៛ៗ៚៙៘,.? ")  # add space
KHNUMBER = list(u"០១២៣៤៥៦៧៨៩0123456789")  # remove 0123456789

CHARS = ["PADDING"] + ["UNK"] + KHCONST + KHVOWEL + KHSUB + KHDIAC + KHSYM + KHNUMBER


chars2idx = {o: i for i, o in enumerate(CHARS)}
idx2chars = {i: o for i, o in enumerate(CHARS)}

train_on_gpu = torch.cuda.is_available()


def cleanup_str(str, separator =""):
    str_ = str.replace("~", separator)
    str_ = str_.replace("^", separator)
    str_ = str_.replace("_", separator)

    return str_


def gen_char_with_label(sentence):
    words = sentence
    final_kccs = []
    for word in words:
        kccs = seg_char(word)
        labels = [1 if (i == 0 or k == " ") else 0 for i, k in enumerate(kccs)]
        final_kccs.extend(list(zip(kccs, labels)))
    return final_kccs


def split_data(X_char, y_char, chars2idx, sentence_length=100):
    X_train_char, X_test_char, y_train_char, y_test_char = train_test_split(
        X_char, y_char, test_size=0.20, random_state=1
    )

    for i, sentence in enumerate(X_train_char):
        # Looking up the mapping dictionary and assigning the index to the respective words
        X_train_char[i] = [chars2idx[c] if c in chars2idx else 1 for c in sentence]

    for i, sentence in enumerate(X_test_char):
        # For test sentences, we have to tokenize the sentences as well
        X_test_char[i] = [chars2idx[c] if c in chars2idx else 1 for c in sentence]

    X_train_char = pad_input(X_train_char, sentence_length)
    X_test_char = pad_input(X_test_char, sentence_length)
    y_train_char = pad_input(y_train_char, sentence_length, False)
    y_test_char = pad_input(y_test_char, sentence_length, False)

    return X_train_char, X_test_char, y_train_char, y_test_char


class WordSegmentRNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden=100, n_layers=2, drop_prob=0.5):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        ## TODO: define the LSTM
        self.lstm = nn.LSTM(
            n_input, n_hidden, n_layers, dropout=drop_prob, batch_first=True
        )

        ## TODO: define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        ## TODO: define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, n_output)

    def forward(self, x, hidden):
        """Forward pass through the network.
        These inputs are x, and the hidden/cell state `hidden`."""

        ## TODO: Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)

        ## TODO: pass through a dropout layer
        out = self.dropout(r_output)

        # Stack up LSTM outputs using view
        # you may need to use contiguous to reshape the output
        out = out.contiguous().view(-1, self.n_hidden)

        ## TODO: put x through the fully-connected layer
        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        """Initializes hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if train_on_gpu:
            hidden = (
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
            )
        else:
            hidden = (
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
            )

        return hidden


def train(
    net, train_dl, test_dl, seq_length=300, epochs=10, lr=0.01, clip=1, print_every=10
):
    """Training a network

    Arguments
    ---------

    net: WordSegmentRNN network
    data: text data to train the network
    epochs: Number of epochs to train
    batch_size: Number of mini-sequences per mini-batch, aka batch size
    seq_length: Number of character steps per mini-batch
    lr: learning rate
    clip: gradient clipping
    val_frac: Fraction of data to hold out for validation
    print_every: Number of steps for printing training and validation loss

    """
    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    if train_on_gpu:
        net.cuda()

    counter = 0
    # n_chars = len(net.chars)
    for e in range(epochs):

        for x, y in train_dl:
            # initialize hidden state
            batch_size = x.shape[0]
            h = net.init_hidden(batch_size)
            counter += 1

            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, len(CHARS))
            inputs, targets = x, y

            if train_on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size * seq_length).long())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss

                val_losses = []
                net.eval()
                for x, y in test_dl:
                    batch_size = x.shape[0]
                    val_h = net.init_hidden(batch_size)
                    # One-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, len(CHARS))
                    inputs, targets = x, y

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y
                    if train_on_gpu:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(
                        output, targets.view(batch_size * seq_length).long()
                    )

                    val_losses.append(val_loss.item())

                net.train()  # reset to train mode after iterationg through validation data

                print(
                    "Epoch: {}/{}...".format(e + 1, epochs),
                    "Step: {}...".format(counter),
                    "Loss: {:.4f}...".format(loss.item()),
                    "Val Loss: {:.4f}".format(np.mean(val_losses)),
                )


def train_model():
    f = open(TRAIN_FILE, "r", encoding="utf-8")

    lines = [cleanup_str(x).split() for x in f]
    char_labels = [gen_char_with_label(sent) for sent in lines]

    chars_only = [[x[0] for x in sent] for sent in char_labels]
    labels_only = [[x[1] for x in sent] for sent in char_labels]

    X_train, X_test, y_train, y_test = split_data(
        chars_only, labels_only, chars2idx, sentence_length=300
    )

    batch_size = 64
    # create your datset
    train_dataset = TensorDataset(
        torch.tensor(X_train).long(), torch.tensor(y_train).long()
    )
    train_dl = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    # create your datset
    test_dataset = TensorDataset(
        torch.tensor(X_test).long(), torch.tensor(y_test).long()
    )
    test_dl = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    model = WordSegmentRNN(len(CHARS), 2)
    train(model, train_dl, test_dl)

    return model


def segment(str):
    if not os.path.exists(MODEL_FILE):
        print("Train on", "gpu" if train_on_gpu else "cpu")
        model = train_model()
        torch.save(model, MODEL_FILE)
    else:
        model = torch.load(MODEL_FILE)

    model.eval()

    list_of_chars = list(str) 

    index_of_chars = [(chars2idx[x] if (x in CHARS) else 1) for x in list_of_chars]

    # print(index_of_chars)

    tensor_chars = torch.from_numpy(np.array(index_of_chars)).unsqueeze(0)
    encoded_chars = one_hot_encode(tensor_chars, len(CHARS))

    if train_on_gpu:
        encoded_chars = encoded_chars.cuda()

    h = model.init_hidden(1)
    h = tuple([each.data for each in h])

    outputs, _ = model(encoded_chars, h)

    if train_on_gpu:
        outputs = outputs.detach().cpu().numpy()
    else:
        outputs = outputs.detach().numpy()


    print(outputs)
    segmented_chars_idx = np.argmax(outputs, axis=1)

    result = ""

    for idx, char in enumerate(list_of_chars):
        if segmented_chars_idx[idx] == 1 and result != "":
            result += " " + char
        else:
            result += char

    return result
