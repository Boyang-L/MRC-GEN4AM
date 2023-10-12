import csv
from sklearn.metrics import classification_report, precision_recall_fscore_support
from options.options import args
from .processing_pe_data_df import data_dev, data_test


def compute_acc(preds_acc):
    with open(args.output_dir + 'prediction.txt', 'w', encoding='utf-8') as o:
        for key, value in preds_acc.items():
            out = key + '\t' + str(value)
            o.write(out)
            o.write('\n')

    acc_labels = []
    truth_acc = []
    label_map_acc = {'MajorClaim': 0, 'Claim': 1, 'Premise': 2}
    if args.do_train:
        input_data = data_dev
    else:
        input_data = data_test

    index = 0
    for line in input_data:
        # todo 这里有问题
        if line['adu_pos'] != '0' and int(line['adu_pos']) != index:
            index = int(line['adu_pos'])
            if str(line['doc_id']) + "\t" + str(line['adu_pos']) in preds_acc.keys():
                ac = preds_acc[str(line['doc_id']) + "\t" + str(line['adu_pos'])]
                acc_labels.append(ac)
            else:
                acc_labels.append(2)
            truth_acc.append(label_map_acc[line['aty']])
    assert len(truth_acc) == len(acc_labels)
    p_acc, r_acc, f1_acc, _ = precision_recall_fscore_support(truth_acc, acc_labels, average='macro')

    clf_report_acc = classification_report(truth_acc, acc_labels, digits=4)
    print('clf_report_acc', clf_report_acc)

    P = p_acc
    R = r_acc
    F1 = f1_acc
    return {
            'precision': P,
            'recall': R,
            'f1_score': F1,
        }


def get_pairs(doc_ids, query_idx, answers):
    preds_acc = {}

    labels_map_acc = {"<MajorClaim>":0 , "<Claim>": 1, "<Premise>": 2}

    for doc_id, query, answer in zip(doc_ids, query_idx, answers):
        ans = answer.split(" ")
        for a in ans:
            if a in labels_map_acc.keys():
                acc = labels_map_acc[a]
                preds_acc[str(doc_id) + "\t" + str(int(query) + 1)] = acc
    return preds_acc 


