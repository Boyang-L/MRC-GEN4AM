import csv
import torch
import os
import collections
import dgl

from torch.utils.data import TensorDataset
from options.options import args
from .processing_pe_data_df import data_train, data_dev, data_test

doc_id_list = []


class InputExample(object):
    def __init__(self, doc_id, context, ac_pos=None, arc=None, acc=None, graph_rep=None):

        self.doc_id = doc_id
        self.context = context
        self.ac_pos = ac_pos
        self.arc = arc
        self.acc = acc
        self.graph_rep = graph_rep


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, decoder_input_ids, acc_label_ids, ard_label_ids, arc_label_ids, query_idx=None, doc_id=None, graph_atten=None, labels_atten=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.decoder_input_ids = decoder_input_ids
        self.acc_label_ids = acc_label_ids
        self.ard_label_ids = ard_label_ids
        self.arc_label_ids = arc_label_ids
        self.query_idx = query_idx
        self.doc_id = doc_id
        self.graph_atten = graph_atten
        self.labels_atten = labels_atten


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class SciArgProcessor(DataProcessor):

    def __init__(self):
        self.ac_labels = ['MajorClaim', 'Claim', 'Premise']
        self.ar_labels = ['none', 'Support', 'Attack']
        self.ac_label_map = self._create_label_map('ac')
        self.ar_label_map = self._create_label_map('ar')

    def _create_label_map(self, label_type):
        label_map = collections.OrderedDict()
        if label_type == 'ac':
            for i, label in enumerate(self.ac_labels):
                label_map[label] = i
        elif label_type == 'ar':
            for i, label in enumerate(self.ar_labels):
                label_map[label] = i
        return label_map

    def convert_labels_to_ids(self, labels, label_type):
        idx_list = []
        if label_type == 'ac':
            for label in labels:
                idx_list.append(self.ac_label_map[label])
        elif label_type == 'ar':
            for label in labels:
                idx_list.append(self.ar_label_map[label])
        return idx_list

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_sci_data(data_train))

    def get_dev_examples(self, data_dir):
        return self._create_examples(self.read_sci_data(data_dev))

    def get_test_examples(self, data_dir):
        return self._create_examples(self.read_sci_data(data_test))

    def get_train_ac_pairs(self, data_dir):
        return self.read_pairs(data_train)

    def get_dev_ac_pairs(self, data_dir):
        return self.read_pairs(data_dev)

    def get_test_ac_pairs(self, data_dir):
        return self.read_pairs(data_test)

    def get_labels(self):
        """ See base class."""
        return self.ac_labels

    @classmethod
    def read_sci_data(cls, input_data):
        dataset = []
        doc_id_pre = 0
        for line in input_data:
            doc_id = line['doc_id']
            if doc_id in doc_id_list:
                continue
            if doc_id != doc_id_pre:
                if doc_id_pre != 0:
                    length = -1
                    for k in data['adu_type'].keys():
                        if len(data['sub_graph'][k]) > length:
                            length = len(data['sub_graph'][k])
                            data['argraph'] = data['sub_graph'][k]
                    for k in data['adu_type'].keys():
                        if k not in data['qapairs'].keys():
                            data['qapairs'][k] = None
                            if data['sub_graph'][k] == []:
                                data['sub_graph'][k] = data['argraph']
                        data['path'][k] = get_path_no_type(data['argraph'], int(data['root_num']), data['all_roots'], int(k))
                    dataset.append(data)
                data = {}
                data['qapairs'] = {}
                data['adu_type'] = {}
                data['sub_graph'] = {}
                data['context'] = []
                data['ac_pos'] = []
                data['acc'] = []
                data['arc'] = []
                data['root'] = None

                data['all_roots'] = []
                data['path'] = {}
                data['root_num'] = -1

                span_pos = -1
                doc_id_pre = doc_id

            data['doc_id'] = str(line['doc_id'])
            if int(line['span_pos']) > span_pos:
                data['context'].append(line['text'])
                span_pos = int(line['span_pos'])

            if line['adu_pos'] != '0':
                if line['adu_pos'] not in data['adu_type'].keys():
                    data['ac_pos'].append(int(line['span_pos']) - 1)
                    data['sub_graph'][str(line['adu_pos'])] = line['rel_pairs']
                    if line['aty'] == 'MajorClaim':
                        data['acc'].append('Claim')
                    else:
                        data['acc'].append(line['aty'])
                    data['arc'].append(str(line['afu']))
                data['adu_type'][str(line['adu_pos'])] = (str(line['aty']), str(line['text']))

            if int(line['root']) == 2:
                data['all_roots'].insert(0, int(line['adu_pos']))

            if int(line['root']) == 1:
                data['all_roots'].append(int(line['adu_pos']))

            # qa_pairs
            if int(line['root']) == 2:
                data['root_num'] = int(line['adu_pos'])
                data['root'] = [
                    [str(line['aty']) + ':' + str(line['afu']), int(line['adu_pos']), str(line['text'])]]
            elif line['parent_pos'] != '0':
                if str(line['parent_pos']) in data['qapairs'].keys():
                    data['qapairs'][str(line['parent_pos'])].append(
                        [str(line['aty']) + ':' + str(line['afu']), int(line['adu_pos']), str(line['text'])])
                else:
                    data['qapairs'][str(line['parent_pos'])] = [
                        [str(line['aty']) + ':' + str(line['afu']), int(line['adu_pos']), str(line['text'])]]
        length = -1
        for k in data['adu_type'].keys():
            if len(data['sub_graph'][k]) > length:
                length = len(data['sub_graph'][k])
                data['argraph'] = data['sub_graph'][k]
        for k in data['adu_type'].keys():
            if k not in data['qapairs'].keys():
                data['qapairs'][k] = None
                if data['sub_graph'][k] == []:
                    data['sub_graph'][k] = data['argraph']
            data['path'][k] = get_path_no_type(data['argraph'], int(data['root_num']), data['all_roots'], int(k))
        dataset.append(data)

        return dataset

    @classmethod
    def read_pairs(cls, input_data):
        adu_pairs = {}
        for line in input_data:
            doc_id = line['doc_id']
            if line['adu_pos'] != '0' and line['parent_pos'] != '0':
                adu1_pos = line['adu_pos']
                adu2_pos = line['parent_pos']
                pairs = doc_id + ' ' + adu1_pos + ' ' + adu2_pos
                rel_label = line['afu']
                adu_pairs[pairs] = rel_label
        return adu_pairs

    # @classmethod
    # def get_pairs_label(cls, input_data):
    #     pairs_label = {}
    #     for line in input_data:
    #         doc_id = line['doc_id']
    #         if line['adu_pos'] != '0' and line['parent_pos'] != '0':
    #             adu1_pos = line['adu_pos']
    #             adu2_pos = line['parent_pos']
    #             p_key = doc_id + ' ' + adu1_pos + ' ' + adu2_pos
    #             pairs_label[p_key] = line['afu']
    #     return pairs_label
    #
    # @classmethod
    # def get_data(cls, input_data):
    #     dataset = []
    #     doc_id_pre = 0
    #     for line in input_data:
    #         doc_id = line['doc_id']
    #         if doc_id != doc_id_pre:
    #             if doc_id_pre != 0:
    #                 dataset.append(data)
    #             data = {}
    #             data['adu_type'] = {}
    #             data['context'] = []
    #             data['ac_pos'] = []
    #             data['acc'] = []
    #             data['arc'] = []
    #             span_pos = -1
    #             doc_id_pre = doc_id
    #
    #         data['doc_id'] = str(line['doc_id'])
    #         if int(line['span_pos']) > span_pos:
    #             data['context'].append(line['text'])
    #             span_pos = int(line['span_pos'])
    #
    #         if line['adu_pos'] != '0':
    #             if line['adu_pos'] not in data['adu_type'].keys():
    #                 data['ac_pos'].append(int(line['span_pos']) - 1)
    #                 new_rel = []
    #                 for r in line['rel_pairs']:
    #                     new_rel.append((int(r[1]), int(r[0])))
    #                 # data['sub_graph'][str(line['adu_pos'])] = new_rel
    #                 if line['aty'] == 'MajorClaim':
    #                     data['acc'].append('Claim')
    #                 else:
    #                     data['acc'].append(line['aty'])
    #                 data['arc'].append(str(line['afu']))
    #             data['adu_type'][str(line['adu_pos'])] = (str(line['aty']), str(line['text']))
    #     dataset.append(data)
    #     return dataset
    #
    # @classmethod
    # def get_ari_both_with_label(cls, dataset, pairs_label):
    #     true_pairs = []
    #     for data in dataset:
    #         doc_id = data['doc_id']
    #         for i in range(len(data['ac_pos'])):
    #             for j in range(len(data['ac_pos'])):
    #                 if i != j:
    #                     p_key = doc_id + ' ' + str(i + 1) + ' ' + str(j + 1)
    #                     if p_key in pairs_label.keys():
    #                         afu = pairs_label[p_key]
    #                     else:
    #                         afu = 'none'
    #                     data_dict = {'doc_id': doc_id,
    #                                  'adu1_pos': i + 1,
    #                                  'adu2_pos': j + 1,
    #                                  'adu1_aty': data['adu_type'][str(i + 1)][0],
    #                                  'adu2_aty': data['adu_type'][str(j + 1)][0],
    #                                  'rel_label': afu}
    #                     true_pairs.append(data_dict)
    #     return true_pairs

    def _create_examples(self, dataset):
        examples = []

        for data in dataset:
            doc_id = data['doc_id']
            context = data['context']
            examples.append(
                InputExample(doc_id, context, ac_pos=data['ac_pos'], arc=data['arc'], acc=data['acc'], graph_rep=data[args.graph_rep_type]))

        return examples

    def convert_examples_to_features(self, examples, tokenizer, max_sen_num,
                                     rel_pairs, max_seq_len=args.max_seq_length,
                                     logger=None,
                                     evaluate=False):
        pad_id = tokenizer.pad_token_id
        query_token = "<query>"
        context_token = "<context>"

        special_tokens = []
        for s_i in range(max_sen_num):
            special_token = "<AC" + str(s_i) + ">"
            special_tokens.append(special_token)

        special_tokens.append("<query>")
        special_tokens.append("<query1>")
        special_tokens.append("<query2>")

        special_tokens.append("<context>")
        special_tokens.append("<non_AC>")
        special_tokens.append("<graph>")
        special_tokens.append("<none>")
        special_tokens.append("<answer>")
        special_tokens.append("<Support>")
        special_tokens.append("<Attack>")
        tokenizer.add_tokens(special_tokens) 
        answer_ids = tokenizer("<answer>").input_ids[1]

        features = []
        for example in examples:

            labels_acc = [0] * args.max_sen_num
            labels_ard = [0] * args.max_sen_num

            # segment
            segments_ids = [1] * max_seq_len

            context_tokens = ""
            for i_c, context in enumerate(example.context):
                if i_c in example.ac_pos:
                    context_tokens = context_tokens + " " + special_tokens[example.ac_pos.index(i_c)] + " " + context
                else:
                    context_tokens = context_tokens + " " + context

            for i in range(len(example.ac_pos)):
                for j in range(len(example.ac_pos)):
                    if i != j:
                        rel_index = example.doc_id + ' ' + str(i + 1) + ' ' + str(j + 1)
                        # arc_label = self.ar_label_map[rel_pairs[rel_index]]
                        # if arc_label != 0:
                        if rel_index in rel_pairs.keys():
                            graph_label = "<graph> "
                            if args.graph_rep_type == "path":
                                for gr in example.graph_rep[str(i + 1)]:
                                    graph_label = graph_label + special_tokens[gr] + " "
                            else:
                                for gr in example.graph_rep[str(j + 1)]:
                                    for gn in gr:
                                        graph_label = graph_label + special_tokens[gn] + " "
                                    graph_label = graph_label + "| "
                                graph_label = graph_label[:-2]
                            if not args.with_graph:
                                graph_label = "<graph> "
                            query = "<query> <AC" + str(i) + "> " + example.context[example.ac_pos[i]] + " <AC" + str(j) + "> " + example.context[example.ac_pos[j]]

                            src_tokens = query + " <context> " + context_tokens
                            src_tokens_idxs = tokenizer(src_tokens, max_length=max_seq_len, return_tensors="pt", pad_to_max_length=True)['input_ids'][0]
                            input_mask = tokenizer(src_tokens, max_length=max_seq_len, return_tensors="pt", pad_to_max_length=True)['attention_mask'][0]
                            assert len(src_tokens_idxs) < max_seq_len + 1

                            labels_arc_t = graph_label + "<answer> <" + rel_pairs[rel_index] + ">"

                            labels_arc_ori = tokenizer(labels_arc_t, max_length=50, return_tensors="pt", pad_to_max_length=True)["input_ids"]
                            labels_arc_ids = labels_arc_ori[:, :-1].contiguous()
                            labels_arc = labels_arc_ori[:, 1:].clone()
                            labels_arc[labels_arc_ori[:, 1:] == pad_id] = -100

                            graph_len = torch.where(labels_arc[0] == answer_ids)[0].tolist()[0]
                            graph_atten = torch.cat((torch.ones(graph_len), torch.zeros(labels_arc.size()[1] - graph_len)))
                            labels_atten = 1 - graph_atten
                            graph_atten = graph_atten.long()
                            labels_atten = labels_atten.long()

                            features.append(
                                InputFeatures(input_ids=src_tokens_idxs.tolist(),
                                              input_mask=input_mask.tolist(),
                                              segment_ids=segments_ids,
                                              decoder_input_ids=labels_arc_ids[0].tolist(),
                                              acc_label_ids=labels_acc,
                                              ard_label_ids=labels_ard,
                                              arc_label_ids=labels_arc[0].tolist(),
                                              query_idx=[i, j],
                                              doc_id=int(example.doc_id),
                                              graph_atten=graph_atten.tolist(),
                                              labels_atten=labels_atten.tolist()))
        return features


    @classmethod
    def features_to_dataset(cls, feature_list, evaluate=False):
        all_input_ids = torch.tensor([f.input_ids for f in feature_list], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature_list], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature_list], dtype=torch.long)
        all_decoder_input_ids = torch.tensor([f.decoder_input_ids for f in feature_list], dtype=torch.long)
        all_query_idx = torch.tensor([f.query_idx for f in feature_list], dtype=torch.long)
        all_doc_id = torch.tensor([f.doc_id for f in feature_list], dtype=torch.long)

        all_acc_label_ids = torch.tensor([f.acc_label_ids for f in feature_list], dtype=torch.long)
        all_ard_label_ids = torch.tensor([f.ard_label_ids for f in feature_list], dtype=torch.long)
        all_arc_label_ids = torch.tensor([f.arc_label_ids for f in feature_list], dtype=torch.long)
        all_graph_atten = torch.tensor([f.graph_atten for f in feature_list], dtype=torch.long)
        all_labels_atten = torch.tensor([f.labels_atten for f in feature_list], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_decoder_input_ids, all_acc_label_ids, all_ard_label_ids,
                                all_arc_label_ids, all_query_idx, all_doc_id, all_graph_atten, all_labels_atten)
        return dataset


def load_and_cache_examples(args, processor, tokenizer, logger, predict=None, evaluate=False):

    logger.info("Creating features from dataset file at %s", args.data_dir)

    if args.do_train:
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir)
            rel_pairs = processor.get_dev_ac_pairs(args.data_dir)
            features = processor.convert_examples_to_features(examples, tokenizer=tokenizer, max_sen_num=args.max_sen_num, evaluate=evaluate, rel_pairs=rel_pairs)
            dataset = processor.features_to_dataset(features, evaluate=evaluate)
            return dataset
        else:
            examples = processor.get_train_examples(args.data_dir)
            rel_pairs = processor.get_train_ac_pairs(args.data_dir)
            features = processor.convert_examples_to_features(examples, tokenizer=tokenizer, max_sen_num=args.max_sen_num,rel_pairs=rel_pairs)
            dataset = processor.features_to_dataset(features)
            return dataset
    else:
        examples = processor.get_test_examples(args.data_dir)
        rel_pairs = processor.get_test_ac_pairs(args.data_dir)
        features = processor.convert_examples_to_features(examples, tokenizer=tokenizer,
                                                          max_sen_num=args.max_sen_num,
                                                          evaluate=evaluate,
                                                          rel_pairs=rel_pairs)
        dataset = processor.features_to_dataset(features, evaluate=evaluate)
        return dataset


def get_path_no_type(old_edges, root, all_roots, node):
    if old_edges == []:
        return [node - 1]
    edges = []
    for edge in old_edges:
        e = (edge[1], edge[0])
        edges.append(e)

    root = root - 1
    node = node - 1
    g = dgl.graph(edges)
    path = []
    for r in all_roots:
        dist, paths = dgl.shortest_dist(g, root=int(r) - 1, return_paths=True)
        if node >= len(dist):
            continue
        if dist[node] == -1:
            continue
        path_node = paths[node].tolist()
        path = []
        for p in path_node:
            if p > -1:
                for n in edges[p]:
                    if n not in path:
                        path.append(n)
    if path == []:
        path = [node]
    return path


def update_embeddings(model, tokenizer):
    embed = model.get_input_embeddings().weight.data

    for i_e in range(args.max_sen_num):
        idx = tokenizer(str(i_e)).input_ids[1]
        extra_idx = tokenizer('<AC' + str(i_e) + '>').input_ids[1]
        embed[extra_idx] = embed[idx]


processors = {
    "sciarg": SciArgProcessor,
}



