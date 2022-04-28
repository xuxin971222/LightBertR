import torch
from torch import nn
from BERT.Bert_model import BERT

class BERTModel(nn.Module):
    def __init__(self, args):
        super(BERTModel, self).__init__()
        self.bert = BERT(args)
        self.prediction = nn.Linear(self.bert.hidden, args.num_items)
        # self.GRU_layer = nn.GRU(128, hidden_size=self.bert.hidden, batch_first=True, num_layers=1)
    def forward(self, x):
        x = self.bert(x)
        # x, hidden = self.GRU_layer(x)
        # x = x[:, -1, :]
        scores = self.prediction(x)
        return scores


