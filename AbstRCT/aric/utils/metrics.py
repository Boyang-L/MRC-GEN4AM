import csv
from sklearn.metrics import classification_report, precision_recall_fscore_support
from options.options import args


def compute_arc(preds_arc):
    with open(args.output_dir + 'prediction.txt', 'w', encoding='utf-8') as o:
        for key, value in preds_arc.items():
            out = key + '\t' + str(value)
            o.write(out)
            o.write('\n')
    arc_labels = []
    truth_arc = []

    label_map_arc = {'none': 0, 'support': 1, 'attack': 2}

    if args.do_train:
        aric_file = '../../data/pairs_both_with_type/neo_dev_pairs_both_with_label.csv'
    else:
        if args.test_set == 'neo':
            aric_file = '../../data/pairs_both_with_type/neo_test_pairs_both_with_label.csv'
        elif args.test_set == 'gla':
            aric_file = '../../data/pairs_both_with_type/gla_test_pairs_both_with_label.csv'
        else:
            aric_file = '../../data/pairs_both_with_type/mix_test_pairs_both_with_label.csv'

    count = 1
    with open(aric_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=",")
        for line in reader:
            truth_arc.append(label_map_arc[line['rel_label']])

            if str(line['doc_id']) + "\t" +  str(line['adu1_pos']) + "\t" + str(line['adu2_pos']) in preds_arc.keys():
                ar = preds_arc[str(line['doc_id']) + "\t" + str(line['adu1_pos']) + "\t" + str(line['adu2_pos'])]
                arc_labels.append(ar)
            else:
                arc_labels.append(0)
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
    length = 0
    not_match = 0
    preds_arc = {}

    labels_map = {}
    labels_map_arc = {'none': 0, "<support>": 1, "<attack>": 2}

    for i in range(args.max_sen_num):
        labels_map["<AC" + str(i) + ">"] = i

    for doc_id, query, answer in zip(doc_ids, query_idx, answers):

        if "<answer>" not in answer or query == -1:
            continue
        answer = answer.split("<answer>")
        if len(answer) < 2:
            continue
        answer = answer[1]
        answer = answer.split(" ")
        a_idx = []
        arc_idx = []
        for i, ans in enumerate(answer):
            if ans in labels_map.keys() and answer[i-1] in labels_map_arc.keys():
                a_idx.append(labels_map[ans])
                arc_idx.append(labels_map_arc[answer[i-1]])
            elif ans in labels_map.keys():
                not_match += 1
            else:
                continue
        length += len(a_idx)
        for a, arc in zip(a_idx, arc_idx):
            if int(a) == int(query):
                continue
            preds_arc[str(doc_id) + "\t" + str(int(a) + 1) + "\t" + str(int(query) + 1)] = arc
    return preds_arc

