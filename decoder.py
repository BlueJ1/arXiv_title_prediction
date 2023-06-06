import torch
import torch.nn as nn
from torchmetrics.text.bert import BERTScore

class LSTMGenerator(nn.Module):
    """
    Class that generates an LSTM model
    TODO: Add Dropout layers?
    TODO: Add more layers?
    :param input_size: The size of the input (embedding)
    :param hidden_size: The number of LSTM units
    :param output_size: The size of the output (vocab size)
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, model_input):
        output, _ = self.lstm(model_input)
        output = self.fc(output[-1])  # Use only the last output
        output = self.softmax(output)
        return output


class BertScoreLoss(nn.Module):
    def __init__(self):
        super(BertScoreLoss, self).__init__()

    def forward(self, y_true, y_pred, bertscore_model=None):
        if bertscore_model is None:
            bertscore_model = BERTScore()

        score = bertscore_model(y_true, y_pred)
        loss = 1 - score['f1']
        return loss