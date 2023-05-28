import torch
import torch.nn as nn
from torchmetrics.text.bert import BERTScore

bertscore = BERTScore()


class LSTMGenerator(nn.Module):
    '''
    Class that generates an LSTM model
    TODO: Add Dropout layers?
    :param input_size: The size of the input (embedding?)
    :param hidden_size: The number of LSTM units
    :param output_size: The size of the output (vocab size?)
    '''

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        output, _ = self.lstm(input)
        output = self.fc(output[-1])  # Use only the last output
        output = self.softmax(output)
        return output


def bert_score_loss(y_true, y_pred):
    '''
    Calculates the loss between the true and predicted values using BERTScore
    The loss is expressed as 1 - f1 score (where f1 is the harmonic mean between precision and recall)

    :param y_true: the true values, here the original titles
    :param y_pred: the predicted values, here the generated titles
    :return: the loss
    '''
    score = bertscore(y_true, y_pred)
    loss = 1 - score['f1']
    return loss
