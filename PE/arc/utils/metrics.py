import csv
from sklearn.metrics import classification_report, precision_recall_fscore_support
from options.options import args
from .processing_pe_data_df import data_dev, data_test


def compute_arc(preds_arc):
    with open(args.output_dir + 'prediction.txt', 'w', encoding='utf-8') as o:
        for key, value in preds_arc.items():
            out = key + '\t' + str(value)
            o.write(out)
            o.write('\n')
    arc_labels = []
    truth_arc = []

    label_map_arc = {'Support': 0, 'Attack': 1}

    if args.do_train:
        input_data = data_dev
    else:
        input_data = data_test
    pairs = get_ari_both_with_label(get_data(input_data), get_pairs_label(input_data))

    for line in pairs:
        if line['rel_label'] not in label_map_arc.keys():
            continue
        truth_arc.append(label_map_arc[line['rel_label']])

        if str(line['doc_id']) + "\t" + str(line['adu1_pos']) + "\t" + str(line['adu2_pos']) in preds_arc.keys():
            ar = preds_arc[str(line['doc_id']) + "\t" + str(line['adu1_pos']) + "\t" + str(line['adu2_pos'])]
            arc_labels.append(ar)
        else:
            arc_labels.append(0)

    assert len(truth_arc) == len(arc_labels)
    p_arc, r_arc, f1_arc, _ = precision_recall_fscore_support(truth_arc, arc_labels, average='macro')

    clf_report_arc = classification_report(truth_arc, arc_labels, digits=4)
    print('clf_report_arc', clf_report_arc)

    P = p_arc
    R = r_arc
    F1 = f1_arc
    return {
            'precision': P,
            'recall': R,
            'f1_score': F1,
        }


def get_pairs(doc_ids, query_idx, answers):
    preds_arc = {}
    labels_map_arc = {"<Support>": 0, "<Attack>": 1}
    for doc_id, query, answer in zip(doc_ids, query_idx, answers):
        if "<answer>" not in answer:
            continue
        answer = answer.split("<answer>")
        if len(answer) < 2:
            print("len<2")
            continue
        answer = answer[1].replace(" ", "")
        if answer in labels_map_arc.keys():
            preds_arc[str(doc_id) + "\t" + str(int(query[0]) + 1) + "\t" + str(int(query[1]) + 1)] = labels_map_arc[answer]
        else:
            continue
    return preds_arc


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