# -*- coding: utf-8 -*-
import time

import numpy as np
import torch
from torch import optim
from transformers.utils import logging
from model import GCNModelVAE, KMeansModel
from optimizer import graph_loss_function, k_means_loss_function
from options import args
from data import load_data
from utils import load_graph_param
from evaluate import single_text_rouge
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    stocks, features, adjs, srcs, tgt = load_data('stock')
    summaries, rouge_list = [], []
    for i in tqdm(range(len(features))):
        stock, feature, adj, src, label = stocks[:i, :-1], features[i], adjs[i], srcs[i], stocks[i, -1]
        n_nodes, feat_dim = feature.shape

        adj_norm, pos_weight, norm, adj_label = load_graph_param(adj)
        gae_model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
        k_means_model = KMeansModel(args.n_clusters)
        optimizer = optim.Adam(gae_model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            gae_model.train()
            optimizer.zero_grad()
            recovered, mu, logvar = gae_model(feature, adj_norm)
            gae_loss = graph_loss_function(preds=recovered, labels=adj_label,
                                           mu=mu, logvar=logvar, n_nodes=n_nodes,
                                           norm=norm, pos_weight=pos_weight)
            cluster_centers, sentence_feat, sentence_index = k_means_model(recovered.detach().numpy().tolist())

            sentence_feat
            k_means_loss = k_means_loss_function(cluster_centers, sentence_feat)
            loss = gae_loss + args.gama * k_means_loss
            loss.backward()
            optimizer.step()
            # cur_loss = loss.item()
            # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
            #       "time=", "{:.5f}".format(time.time() - t)
            #       )

        summary = '.'.join([src[ind] for ind in sentence_index])
        # print(summary)
        summaries.append(summary)

        # supervised summarization
        if tgt:
            res = single_text_rouge(summary, tgt[i])
            rouge_list.append([res[0]["rouge-1"]['f'], res[0]["rouge-2"]['f'], res[0]["rouge-l"]['f']])


    if tgt:
        rouge_score = np.array(rouge_list)
        print("total\n", "rouge-1=", "{:.5f}".format(np.mean(rouge_score[:,0])), "rouge-2=",
              "{:.5f}".format(np.mean(rouge_score[:,1])), "rouge-l=",
              "{:.5f}".format(np.mean(rouge_score[:,2])))


if __name__ == '__main__':
    logging.set_verbosity_error()
    main()
