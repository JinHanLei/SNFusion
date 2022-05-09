# -*- coding: utf-8 -*-
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, PegasusTokenizer, PegasusForConditionalGeneration, PegasusModel, \
    logging
from options import args
import scipy.sparse as sp
logging.set_verbosity_error()


class BertFeature:
    def __init__(self):
        self.device = args.device
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_ch)
        self.model = BertModel.from_pretrained(args.bert_ch).to(self.device)

    def transform(self, texts):
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        result = last_hidden_states.cpu().detach().numpy().tolist()
        return result[0][0]


class PegasusFeature:

    def __init__(self):
        self.device = args.device
        self.tokenizer = PegasusTokenizer.from_pretrained(args.PEGASUS_multi_news)
        self.gen = PegasusForConditionalGeneration.from_pretrained(args.PEGASUS_multi_news).to(self.device)
        self.model = PegasusModel.from_pretrained(args.PEGASUS_multi_news)

    def transform(self, texts):
        inputs = self.tokenizer(texts, truncation=True, padding='longest', return_tensors="pt").input_ids
        if args.PEGASUS_gen:
            trans = self.gen.generate(**inputs)
            tgt_text = self.tokenizer.batch_decode(trans, skip_special_tokens=True)
            return tgt_text
        else:
            decode = inputs.ne(self.model.config.pad_token_id).long()
            output = self.model(input_ids=inputs, decoder_input_ids=decode)
            last_hidden_states = output.last_hidden_state
            result = last_hidden_states.cpu().detach().numpy().tolist()
            return result[0][0]


def preprocess_graph(adj):
    adj = adj + adj.T
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def load_graph_param(adj):
    adj_norm = preprocess_graph(adj)
    pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray())
    return adj_norm, pos_weight, norm, adj_label


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
