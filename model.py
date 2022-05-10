import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.cluster import KMeans
from layers import GraphAttentionLayer, AttentionLayer
from layers import GraphConvolution
# from torch.nn import GRU


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


class MultiHeadAttention(nn.Module):
    def __init__(self, nheads, in_news_dim, in_stock_dim, out_dim, dropout, alpha):
        super(MultiHeadAttention, self).__init__()
        self.attention = [AttentionLayer(in_news_dim, in_stock_dim, out_dim, dropout, alpha) for _ in range(nheads)]
        self.fc = nn.Linear(out_dim * nheads, out_dim)

    def forward(self, news, stock):
        # n_clusters * (out_dim * heads)
        output = torch.cat([atte(news, stock) for atte in self.attention], dim=1).squeeze(0)
        output = F.softmax(self.fc(output), dim=1).mean(dim=0).unsqueeze(0)
        return output


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
            for j, (f, label) in enumerate(zip(feature, labels)):
                if label == i:
                    euc_distance.append([j, np.linalg.norm(center - f), f])
            euc_distance = sorted(euc_distance, key=lambda x: x[1])
            sentence_index.append(euc_distance[0][0])
            sentence_feat.append(euc_distance[0][2])
        cluster_centers = torch.FloatTensor(cluster_centers)
        sentence_feat = torch.FloatTensor(sentence_feat)
        return cluster_centers, sentence_feat, sentence_index


class GRU(nn.Module):
    r"""math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}"""
    def __init__(self, input_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = output_size
        self.cell_size = output_size
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.gate = nn.Linear(input_size + output_size, output_size)

    def forward(self, stock, h_t):
        combined = torch.cat((stock, h_t), 1)
        z = self.sigmoid(self.gate(combined))
        r = self.sigmoid(self.gate(combined))
        h = self.tanh(self.gate(torch.cat((stock, torch.mul(r, h_t)), 1)))
        h = torch.add(torch.mul(z, h), torch.mul(1 - z, h_t))
        return h

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


class FusionModel(nn.Module):
    def __init__(self, text_dim, stock_dim, hidden1, hidden2, n_clusters, gru_dim, n_heads, n_nodes, output_size, dropout, alpha):
        super(FusionModel, self).__init__()
        self.gae_model = GCNModelVAE(text_dim, hidden1, hidden2, dropout)
        self.k_means_model = KMeansModel(n_clusters)
        self.gru_model = GRU(stock_dim, gru_dim)
        self.attention = MultiHeadAttention(n_heads, n_nodes, gru_dim, output_size, dropout, alpha)

    def forward(self, stock, h_t):
        stock_feat = self.gru_model.init_hidden()
        for stock_daily in stock:
            stock_daily = torch.FloatTensor([stock_daily])
            stock_feat = self.gru_model(stock_daily, stock_feat)
        return h