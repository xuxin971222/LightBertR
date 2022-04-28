import torch
from torch.utils.data import Dataset

class BertDataset(Dataset):
    def __init__(self, data,  max_len_session,  mask_prob, mask_token, num_items, rng):
        self.session = data
        # self.neigh_session = neigh_s
        self.max_len_session = max_len_session
        # self.max_len_neigh = max_len_neigh
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.session)

    def __getitem__(self, index):
        tokens = []
        labels = []
        for s in self.session[index][:-1]:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(0, self.num_items))
                else:
                    tokens.append(s)
                labels.append(s)
            else:
                tokens.append(s)
                labels.append(-1)

        tokens.append(self.mask_token)
        labels.append(-1)

        tokens = tokens[-self.max_len_session:]
        labels = labels[-self.max_len_session:]
        # neighs = self.neigh_session[index][-self.max_len_neigh:]

        mask_len_s = self.max_len_session - len(tokens)
        # mask_len_n = self.max_len_neigh - len(neighs)
        # tokens = [self.mask_token + 1] * mask_len_s + tokens
        # labels = [self.mask_token + 1] * mask_len_s + labels

        tokens = [self.mask_token] * mask_len_s + tokens
        labels = [-1] * mask_len_s + labels
        # neighs = [self.mask_token + 1] * mask_len_n + neighs
        return torch.LongTensor(tokens), torch.LongTensor(labels)


# self.ce = nn.CrossEntropyLoss(ignore_index=0)