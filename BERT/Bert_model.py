import torch
from torch import nn
from BERT.Bert_embedding import BERTEmbedding
from BERT.TransformBlock import TransformerBlock
from BERT.random_seed import fix_random_seed_as
from BERT.Filter_block import FilterLayer
from BERT.res_net import LayerNorm


class BERT(nn.Module):
    def __init__(self, args):
        super(BERT, self).__init__()
        fix_random_seed_as()
        self.max_len = args.bert_max_len
        self.n_layers = args.bert_num_blocks
        self.heads = args.bert_num_heads
        self.hidden = args.bert_hidden_units
        self.dropout = args.bert_dropout
        self.vocab_size = args.num_items + 1

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=self.vocab_size, embed_size=self.hidden, max_len=self.max_len, dropout=self.dropout)
        self.linear = nn.Linear(self.hidden, 51)
        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden, self.heads, self.hidden * 4, self.dropout) for _ in range(self.n_layers)])
        # self.init_weights()

        # Filter_Block
        self.dense = nn.Linear(self.hidden, self.hidden)
        self.LayerNorm = LayerNorm(self.hidden, eps=1e-12)
        self.out_dropout = nn.Dropout(self.dropout)

        self.filter_block = FilterLayer(args)

    def forward(self, x):
        mask = (x < self.vocab_size - 1).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)
        # filter
        x = self.filter_block(x)
        # running over multiple transformer blocks
        # for transformer in self.transformer_blocks:
        #     x = transformer.forward(x, mask)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        # x = self.LayerNorm(self.out_dropout(self.dense(x)))

        return x

    def init_weights(self):
        pass






















