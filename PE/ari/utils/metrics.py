import csv
from sklearn.metrics import classification_report, precision_recall_fscore_support
from options.options import args
from .processing_pe_data_df import data_dev, data_test


def compute_ari(preds):
    preds_ard = preds
    preds_ard = set(preds_ard)
    preds_ard = list(preds_ard)
    preds_ard = sorted(preds_ard)

    with open(args.output_dir + 'prediction.txt', 'w', encoding='utf-8') as o:
        for pre in preds_ard:
            out = '\t'.join(str(p) for p in pre)
            o.write(out)
            o.write('\n')
    ard_labels = []
    truth_ard = []

    label_map_ard = {'none': 0, 'Support': 1, 'Attack': 1}

    if args.do_train:
        input_data = data_dev
    else:
        input_data = data_test
    pairs = get_ari_both_with_label(get_data(input_data), get_pairs_label(input_data))

    for line in pairs:
        true_rel = (str(line['doc_id']), int(line['adu1_pos']), int(line['adu2_pos']))
        truth_ard.append(label_map_ard[line['rel_label']])
        if true_rel in preds_ard:
            ard_labels.append(1)
        else:
            ard_labels.append(0)
    assert len(truth_ard) == len(ard_labels)
    p_ard, r_ard, f1_ard,_ = precision_recall_fscore_support(truth_ard, ard_labels, average='macro')

    clf_report_ard = classification_report(truth_ard, ard_labels, digits=4)
    print('clf_report_ard', clf_report_ard)

    P = p_ard
    R = r_ard
    F1 = f1_ard 
    return {
            'precision': P,
            'recall': R,
            'f1_score': F1,
        }


def get_pairs(doc_ids, query_idx, answers):
    length = 0
    preds = []
    labels_map = {}
    for i in range(args.max_sen_num):
        labels_map["<AC" + str(i) + ">"] = i

    for doc_id, query, answer in zip(doc_ids, query_idx, answers):
        if "<answer>" not in answer or query == -1:
            continue
        answer = answer.split("<answer>")[1]
        answer = answer.split(" ")
        a_idx = []
        for ans in answer:
            if ans in labels_map.keys():
                a_idx.append(labels_map[ans])
            else:
                continue
        length += len(a_idx)
        for a in a_idx:
            if int(a) == int(query):
                continue
            preds.append((str(doc_id), int(a) + 1, int(query) + 1))
    return preds


def get_pairs_label(input_data):
    pairs_label = {}
    for line in input_data:
        doc_id = line['doc_id']
        if line['adu_pos'] != '0' and line['parent_pos'] != '0':
            adu1_pos = line['adu_pos']
            adu2_pos = line['parent_pos']
            p_key = doc_id + ' ' + adu1_pos + ' ' + adu2_pos
            pairs_label[p_key] = line['afu']
    return pairs_label


def get_data(input_data):
    dataset = []
    doc_id_pre = 0
    for line in input_data:
        doc_id = line['doc_id']
        if doc_id != doc_id_pre:
            if doc_id_pre != 0:
                dataset.append(data)
            data = {}
            data['adu_type'] = {}
            data['context'] = []
            data['ac_pos'] = []
            data['acc'] = []
            data['arc'] = []
            span_pos = -1
            doc_id_pre = doc_id

        data['doc_id'] = str(line['doc_id'])
        if int(line['span_pos']) > span_pos:
            data['context'].append(line['text'])
            span_pos = int(line['span_pos'])

        if line['adu_pos'] != '0':
            if line['adu_pos'] not in data['adu_type'].keys():
                data['ac_pos'].append(int(line['span_pos']) - 1)
                new_rel = []
                for r in line['rel_pairs']:
                    new_rel.append((int(r[1]), int(r[0])))
                # data['sub_graph'][str(line['adu_pos'])] = new_rel
                if line['aty'] == 'MajorClaim':
                    data['acc'].append('Claim')
                else:
                    data['acc'].append(line['aty'])
                data['arc'].append(str(line['afu']))
            data['adu_type'][str(line['adu_pos'])] = (str(line['aty']), str(line['text']))
    dataset.append(data)
    return dataset


def get_ari_both_with_label(dataset, pairs_label):
    true_pairs = []
    for data in dataset:
        doc_id = data['doc_id']
        for i in range(len(data['ac_pos'])):
            for j in range(len(data['ac_pos'])):
                if i != j:
                    p_key = doc_id + ' ' + str(i + 1) + ' ' + str(j + 1)
                    if p_key in pairs_label.keys():
                        afu = pairs_label[p_key]
                    else:
                        afu = 'none'
                    data_dict = {'doc_id': doc_id,
                                 'adu1_pos': i + 1,
                                 'adu2_pos': j + 1,
                                 'adu1_aty': data['adu_type'][str(i + 1)][0],
                                 'adu2_aty': data['adu_type'][str(j + 1)][0],
                                 'rel_label': afu}
                    true_pairs.append(data_dict)
    return true_pairs
