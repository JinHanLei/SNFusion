import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.cluster import KMeans
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from layers import GraphConvolution


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class KMeansModel(nn.Module):
    def __init__(self, n_clusters=10, random_state=666):
        super(KMeansModel, self).__init__()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.k_means_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

    def forward(self, feature):
        k_means = self.k_means_model.fit(feature)
        cluster_centers = k_means.cluster_centers_
        labels = k_means.labels_
        # 簇内距离之和
        # inertia = k_means.inertia_
        sentence_index, sentence_feat = [], []
        for i, center in enumerate(cluster_centers):
            euc_distance = []
            for j, f, label in enumerate(zip(feature, labels)):
                if label == i:
                    euc_distance.append([j, np.linalg.norm(center - f), f])
            euc_distance = sorted(euc_distance, key=lambda x: x[1])
            sentence_index.append(euc_distance[0][0])
            sentence_feat.append(euc_distance[0][2])
        cluster_centers = torch.FloatTensor(cluster_centers)
        sentence_feat = torch.FloatTensor(sentence_feat)
        return cluster_centers, sentence_feat, sentence_index


class GRU(nn.Module):
    def __init__(self, input_size, output_size, num_classes=2, classification=False):
        super(GRU, self).__init__()
        self.hidden_size = output_size
        self.cell_size = output_size
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.gate = nn.Linear(input_size + output_size, output_size)
        self.classification = classification
        if self.classification:
            self.output_dense = nn.Linear(output_size, num_classes)

    def forward(self, input, h_t):
        combined = torch.cat((input, h_t), 1)

        z = self.sigmoid(self.gate(combined))
        r = self.sigmoid(self.gate(combined))
        h_m = self.tanh(self.gate(torch.cat((input, torch.mul(r, h_t)), 1)))
        h = torch.add(torch.mul(z, h_m), torch.mul(1 - z, h_t))

        if self.classification:
            output = self.output_dense(h)
        else:
            output = h
        return output, h

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


