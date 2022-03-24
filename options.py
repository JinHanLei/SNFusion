import argparse

parser = argparse.ArgumentParser(description='Args for FinGC.')

# PATH
parser.add_argument('--multi_news', type=str, default="./data/multi-news/", help='Root path of data.')
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--BERT_EN', type=str, default="./ckpts/bert_en")
parser.add_argument('--pre_train_model', type=str, default="BERT")
parser.add_argument('--PEGASUS_multi_news', type=str, default="./ckpts/pegasus/multi_news")
parser.add_argument('--PEGASUS_gen', type=bool, default=False)
parser.add_argument('--sim_threshold', type=float, default=0.6)


parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=512, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=128, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--n_clusters', type=float, default=3)
parser.add_argument('--gama', type=float, default=1)

args = parser.parse_args()