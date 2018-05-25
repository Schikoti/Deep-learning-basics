"""
Run similar model as in lab 4

modified from https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import plot_confusion_matrix
import matplotlib.pyplot as plt
from torchtext.data import Field
from torchtext.data import BucketIterator
from torchtext.data import TabularDataset
import torch.nn as nn

DATA_ROOT = "./datasets/"


def load_dataset(db_name, batch_size):
    """
    Load the csv datasets into torchtext files

    Inputs:
    db_name (string)
       The name of the dataset. This name must correspond to the folder name.
    batch_size
       The batch size
    """
    print "Loading " + db_name + "..."
    i = 1
    print('num', i)

    tokenize = lambda x: x.split()
    TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
    LABEL = Field(sequential=False, use_vocab=False)

    tv_datafields = [("sentence", TEXT),
                     ("label", LABEL)]

    trn, vld = TabularDataset.splits(
        path=DATA_ROOT + db_name,  # the root directory where the data lies
        train='train.csv', validation="test.csv",
        format='csv',
        skip_header=False,
        fields=tv_datafields)

    TEXT.build_vocab(trn)

    print "vocab size: %i" % len(TEXT.vocab)

    train_iter, val_iter = BucketIterator.splits(
        (trn, vld),
        batch_sizes=(batch_size, batch_size),
        device=-1,  # specify dont use gpu
        sort_key=lambda x: len(x.sentence),  # sort the sentences by length
        sort_within_batch=False,
        repeat=False)

    return train_iter, val_iter, len(TEXT.vocab)


hidden_size = 20  # output_size
embed_size = 3
num_layers_rnn = 1
hidden_dim = 6
target_size = 2
batch_size = 8

train_iterator, test_iterator, vocab_size = load_dataset("sentiment", batch_size)
GPU=torch.cuda.is_available()


class BaseNet(nn.Module):
    def __init__(self, embed_size, hidden_dim, vocab_size, target_size, batch_size):
        super(BaseNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_dim)
        self.FC = nn.Linear(hidden_dim, target_size)

    def init_hidden(self):
        if GPU:
            hidden=(Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).cuda(),
                Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).cuda())
        else:
            hidden=(Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
             Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

        return hidden

    def forward(self, sentence):
        if GPU:
            sentence=sentence.cuda()
        embeds = self.word_embeddings(sentence)
        lstm_out, (hn, cn) = self.lstm(embeds)
        fc_out = self.FC(lstm_out[-1, :, :])
        return fc_out

    def fit(self, train_iterator):
        # switch to train mode
        self.train()

        # define loss function
        criterion = nn.CrossEntropyLoss()
        criterion= criterion.cuda()
        # setup SGD
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        for epoch in range(20):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(train_iterator, 0):
                # get the inputs
                sentence = data.sentence
                label = data.label

                if GPU:
                    sentence=sentence.cuda()
                    label=label.cuda()

                # wrap them in Variable
                # inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                # optimizer.zero_grad()

                self.zero_grad()
                self.hidden = self.init_hidden()

                # compute forward pass
                outputs = self.forward(sentence)

                # get loss function
                loss = criterion(outputs, label)

                # do backward pass
                loss.backward()

                # do one gradient step
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]
            print('Epoch', epoch)
            print('[Epoch: %d] loss: %.3f' %
                  (epoch + 1, running_loss / (i + 1)))
            running_loss = 0.0
            self.predict(test_iterator)
        print('Finished Training')

    def predict(self, test_iterator):
        # switch to evaluate mode
        self.eval()

        correct = 0
        total = 0
        all_predicted = []
        orig_labels = []
        for data in test_iterator:
            sentence = data.sentence
            label = data.label
            if GPU:
                sentence=sentence.cuda()
            outputs = self.forward(sentence)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (Variable(predicted).data.cpu().numpy() == label.data.cpu().numpy()).sum()
            all_predicted += predicted.tolist()
            orig_labels += label.data.cpu().numpy().tolist()
        print('Accuracy: %d %%' % (
                100 * correct / total))

        return all_predicted, orig_labels


def main():
    model = BaseNet(embed_size, hidden_dim, vocab_size, target_size, batch_size)
    if GPU:
        model=model.cuda()
    model.fit(train_iterator)
    pred_labels, test_labels = model.predict(test_iterator)
    plt.figure(1)
    plot_confusion_matrix(pred_labels, test_labels, "LSTM")
    plt.show()
    plt.savefig('Confusion Matrix_LSTM_news')
    torch.save(model, 'pytorchpart2_cudanews.pt')



if __name__ == "__main__":
    main()
