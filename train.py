import os
import torch
from time import ctime
from model import BERTModel
from torch import nn, optim
from metrics import Metrics
from tqdm import tqdm

# GMF_model = GMF.GMF(user_num, item_num, args.embedding_dim_GMF, args.dropout)
# GMF_model.load_state_dict(torch.load(GMF_model_path))

def train(data_info, args):
    loader_train = data_info['bert_loader_train']
    loader_test = data_info['bert_loader_test']
    lr = data_info['lr']
    epochs = data_info['epochs']
    top_k = data_info['top_k']

    model = BERTModel(args).cuda()
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), weight_decay=1e-5, lr=lr) # weight_decay:1e-5, lr:1e-4
                                                                                             # 0.01
    #### model load
    # Bert_model_path = os.path.join(args.bert_save_path, 'bert.pth')
    # model.load_state_dict(torch.load(Bert_model_path))
    ####

    print("training....", ctime())
    best_ndcg= 0; best_hr = 0; best_mrr = 0
    for epoch in range(epochs):
        model.train()
        loss_total = 0
        for data, target in tqdm(loader_train):
            data = data.cuda()
            scores = model(data)

            targets = target.view(-1)
            scores = scores.view(-1, scores.size(-1))  # (B*T) x V
            scores = scores.cpu()

            loss = loss_function(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        print(epoch, loss_total, ctime())

        model.eval()
        i = 0
        HR, NDCG, MRR = Metrics(model, loader_test, top_k)
        print("HR@{}:{}".format(top_k[i], HR[i]), "NDCG@{}:{}".format(top_k[i], NDCG[i]), "MRR@{}:{}".format(top_k[i], MRR[i]))

        if HR[i] > best_hr:
           best_hr = HR[i]
           torch.save(model.state_dict(), os.path.join(args.bert_save_path, 'bert.pth'))

        best_mrr = MRR[i] if MRR[i] > best_mrr else best_mrr
        best_ndcg = NDCG[i] if NDCG[i] > best_ndcg else best_ndcg

        print("best_HR@{}:{}".format(top_k[i], best_hr), "best_NDCG@{}:{}".format(top_k[i], best_ndcg), "bestMRR@{}:{}".format(top_k[i], best_mrr))




