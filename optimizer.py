import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import torch.nn as nn


def graph_loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


def k_means_loss_function(x, y):
    criterion = nn.KLDivLoss(reduction='sum')
    assert len(x) == len(y), '聚类数量不一致！'
    klloss = 0
    for i in range(len(x)):
        logp_x = F.log_softmax(x[i], dim=-1)
        p_y = F.softmax(y[i], dim=-1)
        klloss += criterion(logp_x, p_y)
    return klloss