import argparse

parser = argparse.ArgumentParser(description='Args for FirmAsGraph.')

# PATH
parser.add_argument('--multi_news', type=str, default="./data/multi-news/", help='Root path of data.')
parser.add_argument('--stock', type=str, default="./data/stock/", help='Root path of data.')
parser.add_argument('--device', type=str, default="cuda:0")
# pre train
parser.add_argument('--bert_en', type=str, default="./ckpts/bert_en")
parser.add_argument('--bert_ch', type=str, default="./ckpts/bert_ch")
parser.add_argument('--sim_threshold', type=float, default=0.6)
# model
parser.add_argument('--model', type=str, default="fusion", help='Output which part of the model.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train.')
# GAE
parser.add_argument('--hidden1', type=int, default=32, help='Number of hidden layer 1 of GAE.')
parser.add_argument('--hidden2', type=int, default=8, help='Number of hidden layer 2 of GAE.')
# GAT
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head GAT.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
# Cluster
parser.add_argument('--n_clusters', type=float, default=3)
# GRU
parser.add_argument('--gru_dim', type=float, default=8)
# Attention
parser.add_argument('--n_heads', type=int, default=4, help='Number of head attentions.')
# optimizer
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
# joint learning
parser.add_argument('--gama', type=float, default=1)
parser.add_argument('--eta', type=float, default=1)
# output
parser.add_argument('--output_size', type=float, default=2)

args = parser.parse_args()