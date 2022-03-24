import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from transformers.utils import logging

from model import GCNModelVAE, KMeansModel
from optimizer import gae_loss_function, k_means_loss_function
from options import args
from utils import load_graph, preprocess_graph
from evaluate import single_text_rouge


def main():
    features, srcs, adjs, tgt = load_graph(args.multi_news)
    summaries = []
    rouge1_list = []
    rouge2_list = []
    rougel_list = []
    for i in range(len(features)):
        feature = torch.FloatTensor(features[i])
        n_nodes, feat_dim = feature.shape
        # model输入
        adj = adjs[i]
        adj_norm = preprocess_graph(adj)
        src = srcs[i]
        # loss所需参数
        pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())

        gae_model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
        k_means_model = KMeansModel(args.n_clusters)
        optimizer = optim.Adam(gae_model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            t = time.time()
            gae_model.train()
            optimizer.zero_grad()
            recovered, mu, logvar = gae_model(feature, adj_norm)
            gae_loss = gae_loss_function(preds=recovered, labels=adj_label,
                                         mu=mu, logvar=logvar, n_nodes=n_nodes,
                                         norm=norm, pos_weight=pos_weight)
            cluster_centers, sentence_feat, sentence_index = k_means_model(recovered.detach().numpy().tolist())
            k_means_loss = k_means_loss_function(cluster_centers, sentence_feat)
            loss = gae_loss + args.gama * k_means_loss
            loss.backward()
            optimizer.step()
            # cur_loss = loss.item()
            #
            # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
            #       "time=", "{:.5f}".format(time.time() - t)
            #       )
        summary_index = [src[ind] for ind in sentence_index]
        summary = '.'.join(summary_index)
        # print(summary, "\t", tgt[i])
        summaries.append(summary)
        res = single_text_rouge(summary, tgt[i])
        rouge1 = res[0]["rouge-1"]['f']
        rouge1_list.append(rouge1)
        rouge2 = res[0]["rouge-2"]['f']
        rouge2_list.append(rouge2)
        rougel = res[0]["rouge-l"]['f']
        rougel_list.append(rougel)
        print("sentence {}\t".format(i), "rouge-1=", "{:.5f}".format(rouge1), "rouge-2=", "{:.5f}".format(rouge2),
              "rouge-l=", "{:.5f}".format(rougel))
    print("total\n", "rouge-1=", "{:.5f}".format(np.mean(rouge1_list)), "rouge-2=",
          "{:.5f}".format(np.mean(rouge2_list)), "rouge-l=",
          "{:.5f}".format(np.mean(rougel_list)))


if __name__ == '__main__':
    logging.set_verbosity_error()
    main()
