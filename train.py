# -*- coding: utf-8 -*-
import time

import numpy as np
import torch
from torch import optim
from transformers.utils import logging
from model import GCNModelVAE, KMeansModel, GRU, MultiHeadAttention
from optimizer import graph_loss_function, k_means_loss_function
from options import args
from data import load_data
from utils import load_graph_param
from evaluate import single_text_rouge
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, recall_score


def main():
    print("loading data...")
    stocks, news, adjs, srcs, tgt = load_data('stock')
    summaries, rouge_list, preds = [], [], []

    print("Training...")
    for day in range(len(news)):
        stock, text, adj, src, label = stocks[:day+1, :-1], news[day], adjs[day], srcs[day], stocks[day, -1]
        print("第{}天是将要{}的".format(day+1, '涨' if label else "跌"))
        text = torch.FloatTensor(text)
        label = torch.LongTensor([label])
        n_nodes, text_dim = text.shape
        stock_dim = len(stock[0])

        adj_norm, pos_weight, norm, adj_label = load_graph_param(adj)

        gae_model = GCNModelVAE(text_dim, args.hidden1, args.hidden2, args.dropout)
        k_means_model = KMeansModel(args.n_clusters)
        gru_model = GRU(stock_dim, args.gru_dim)
        fusion_model = MultiHeadAttention(args.n_heads, n_nodes, args.gru_dim, args.output_size, args.dropout, args.alpha)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(gae_model.parameters(), lr=args.lr)
        out, sentence_index = None, None
        for epoch in tqdm(range(args.epochs)):
            gae_model.train()
            optimizer.zero_grad()
            # GAE
            recovered, mu, logvar = gae_model(text, adj_norm)
            gae_loss = graph_loss_function(preds=recovered, labels=adj_label,
                                           mu=mu, logvar=logvar, n_nodes=n_nodes,
                                           norm=norm, pos_weight=pos_weight)
            # K-Means
            cluster_centers, sentence_feat, sentence_index = k_means_model(recovered.detach().numpy().tolist())
            k_means_loss = k_means_loss_function(cluster_centers, sentence_feat)
            # GRU
            stock_feat = gru_model.init_hidden()
            for stock_daily in stock:
                stock_daily = torch.FloatTensor([stock_daily])
                stock_feat = gru_model(stock_daily, stock_feat)
            # Fusion
            out = fusion_model(sentence_feat, stock_feat)
            fusion_loss = criterion(out, label)

            loss = gae_loss + args.gama * k_means_loss + args.eta * fusion_loss
            loss.backward()
            optimizer.step()
            # cur_loss = loss.item()
            # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
            #       "time=", "{:.5f}".format(time.time() - t))

        _, out_binary = torch.max(out, 1)
        preds.append(out_binary.cpu().tolist()[0])

        summary = ''.join([src[ind].strip() for ind in sentence_index])
        # print(summary)
        summaries.append(summary)

        # supervised summarization
        if tgt:
            res = single_text_rouge(summary, tgt[day])
            rouge_list.append([res[0]["rouge-1"]['f'], res[0]["rouge-2"]['f'], res[0]["rouge-l"]['f']])
    if tgt:
        rouge_score = np.array(rouge_list)
        print("total\n", "rouge-1=", "{:.5f}".format(np.mean(rouge_score[:,0])), "rouge-2=",
              "{:.5f}".format(np.mean(rouge_score[:,1])), "rouge-l=",
              "{:.5f}".format(np.mean(rouge_score[:,2])))
    print(preds)
    print(summaries)
    print(accuracy_score(stocks[:, -1], preds))


if __name__ == '__main__':
    logging.set_verbosity_error()
    main()