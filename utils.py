# -*- coding: utf-8 -*-
import itertools
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, PegasusTokenizer, PegasusForConditionalGeneration, PegasusModel, \
    logging
from options import args
import scipy.sparse as sp
import pickle as pkl
import re


class MultiNewsSummaryDataset:

    @classmethod
    def parse_source_data(cls, content):
        # content_transform = re.sub('(\$.*?M\s)|(#.*?\s)|(\|\|\|\|\|)|(\d+\.\d+%\s?)|(\d+\.\d+\s?)', '', content)
        content_transform = re.sub('\|\|\|\|\||NEWLINE_CHAR', '', content)
        sentence_split_list = cls.sentence_split_en(content_transform)
        mark, feature_list = cls.pre_train(sentence_split_list)
        assert len(sentence_split_list) == len(feature_list), (len(sentence_split_list),len(feature_list))
        if len(mark) != 0:
            sentence_split_list = [sentence_split_list[i] for i in range(len(sentence_split_list)) if (i not in mark)]
        l = len(feature_list)
        adj = np.zeros((l, l))
        src_index = range(l)
        index_combinations = list(itertools.combinations(src_index, 2))
        for i, j in index_combinations:
            cos_sim = torch.cosine_similarity(torch.Tensor(feature_list[i]), torch.Tensor(feature_list[j]), dim=0)
            if cos_sim >= args.sim_threshold:
                adj[i][j] = cos_sim.numpy()
        adj = sp.csr_matrix(adj)
        return sentence_split_list, feature_list, adj

    @classmethod
    def pre_train(cls, sent_list):
        feature_list = []
        if args.pre_train_model == 'BERT':
            pre_model = BertFeature()
        elif args.pre_train_model == 'PEGASUS':
            pre_model = PegasusFeature()
        else:
            assert "Undefined pre-training model"
        mark_unused_sen = -1
        unused_sen = []
        for sen in sent_list:
            mark_unused_sen += 1
            try:
                feature = pre_model.transform(sen)
                feature_list.append(feature)
            except:
                unused_sen.append(mark_unused_sen)
                print(sen)
                continue
        return unused_sen, feature_list

    @staticmethod
    def sentence_split_en(str_sentence):
        list_ret = list()
        for s_str in str_sentence.split('. '):
            if '? ' in s_str:
                list_ret.extend(s_str.split('? '))
            elif '! ' in s_str:
                list_ret.extend(s_str.split('! '))
            else:
                list_ret.append(s_str)
        # for s_str in list_ret:
        #     space = s_str.count(" ")
        #     if space > 509:
        #         # list_final.append(' '.join(s_str.split(' ')[:100]))
        #         continue
        #     if space > 5:
        #         list_final.append(s_str.strip())
        # 过滤短句
        list_ret = [x.strip() for x in list_ret if x.count(" ") > 5]
        return list_ret

    @classmethod
    def load_data(cls, data_path, mode="train"):
        """
        mode : {"train", "val", "test"}
        """
        source_data_path = os.path.join(data_path, "{}.src".format(mode))
        target_data_path = os.path.join(data_path, "{}.tgt".format(mode))
        source_data = []
        source_data_feature = []
        graph = []
        target_data = []
        source_pro_path = "./res/{}_{}_sen.pkl".format(args.pre_train_model, mode)
        feature_path = "./res/{}_{}_feat.pkl".format(args.pre_train_model, mode)
        graph_path = "./res/{}_{}_graph.pkl".format(args.pre_train_model, mode)
        count = 0
        if os.path.exists(feature_path):
            source_data = pkl.load(open(source_pro_path, "rb"))
            source_data_feature = pkl.load(open(feature_path, "rb"))
            graph = pkl.load(open(graph_path, "rb"))
        else:
            with open(source_data_path, encoding='UTF-8') as f_src:
                for lines in tqdm(f_src.readlines()):
                    src, feat, adj = cls.parse_source_data(lines)
                    source_data.append(src)
                    source_data_feature.append(feat)
                    graph.append(adj)
                    count += 1
                    if count > 100:
                        break
                with open(source_pro_path, 'wb') as f:
                    pkl.dump(source_data, f)
                with open(graph_path, 'wb') as f:
                    pkl.dump(graph, f)
                with open(feature_path, 'wb') as f_features:
                    pkl.dump(source_data_feature, f_features)
        with open(target_data_path, encoding='UTF-8') as f_tgt:
            for line in f_tgt.readlines():
                target_data.append(line.strip())
        return source_data_feature, source_data, graph, target_data


class BertFeature:

    def __init__(self):
        self.device = args.device
        self.tokenizer = BertTokenizer.from_pretrained(args.BERT_EN)
        self.model = BertModel.from_pretrained(args.BERT_EN).to(self.device)

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


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_graph(datasets):
    # bert特征，句子文本，邻接矩阵，标签句子
    feat, src, graph, tgt = MultiNewsSummaryDataset.load_data(datasets, mode="test")
    return feat, src, graph, tgt


if __name__ == "__main__":
    logging.set_verbosity_error()
    load_graph()
