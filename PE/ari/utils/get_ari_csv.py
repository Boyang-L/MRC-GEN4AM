import csv

doc_id_list = []


def read_pairs_label(input_file):
    with open(input_file, "r") as f:
        pairs_label = {}
        reader = csv.DictReader(f, delimiter=",")
        for line in reader:
            doc_id = line['doc_id']
            if line['adu_pos'] != '0' and line['parent_pos'] != '0':
                adu1_pos = line['adu_pos']
                adu2_pos = line['parent_pos']
                p_key = doc_id + ' ' + adu1_pos + ' ' + adu2_pos
                pairs_label[p_key] = line['afu']
    return pairs_label


def read_csv(input_file):
    with open(input_file, "r") as f:
        dataset = []
        reader = csv.DictReader(f, delimiter=",")
        doc_id_pre = 0
        for line in reader:
            doc_id = line['doc_id']
            if doc_id in doc_id_list:
                continue
            if doc_id != doc_id_pre:
                if doc_id_pre != 0:
                    length = -1

                    for k in data['sub_graph'].keys():
                        if len(data['sub_graph'][k]) > length:
                            length = len(data['sub_graph'][k])
                            data['argraph'] = data['sub_graph'][k]
                    dataset.append(data)
                data = {}
                data['adu_type'] = {}
                data['sub_graph'] = {}
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
                    for r in eval(line['rel_pairs']):
                        new_rel.append((int(r[1]), int(r[0])))
                    data['sub_graph'][str(line['adu_pos'])] = new_rel
                    if line['aty'] == 'MajorClaim':
                        data['acc'].append('Claim')
                    else:
                        data['acc'].append(line['aty'])
                    data['arc'].append(str(line['afu']))
                data['adu_type'][str(line['adu_pos'])] = (str(line['aty']), str(line['text']))
        length = -1
        for k in data['sub_graph'].keys():
            if len(data['sub_graph'][k]) > length:
                length = len(data['sub_graph'][k])
                data['argraph'] = data['sub_graph'][k]
        dataset.append(data)
    return dataset


def get_ari_both_with_label(dataset, pairs_label):
    output_csv = open('./data/' + settype + '_pairs_both_with_label.csv', 'w', newline='', encoding='utf-8')
    fieldnames = ['doc_id', 'adu1_pos', 'adu2_pos', 'adu1_aty', 'adu2_aty', 'rel_label', 'adu1_text', 'adu2_text']
    writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
    writer.writeheader()

    for data in dataset:
        doc_id = data['doc_id']
        for i in range(len(data['ac_pos'])):
            for j in range(len(data['ac_pos'])):
                if i != j:
                    p_key = doc_id + ' ' + str(i + 1) + ' ' + str(j + 1)
                    if p_key in pairs_label.keys():
                        afu = paris_label[p_key]
                    else:
                        afu = 'none'
                    write_dict = {'doc_id': doc_id,
                                  'adu1_pos': i + 1,
                                  'adu2_pos': j + 1,
                                  'adu1_aty': data['adu_type'][str(i + 1)][0],
                                  'adu2_aty': data['adu_type'][str(j + 1)][0],
                                  'rel_label': afu,
                                  'adu1_text': data['adu_type'][str(i + 1)][1],
                                  'adu2_text': data['adu_type'][str(j + 1)][1]}
                    writer.writerow(write_dict)
    return


if __name__ == '__main__':
    for settype in ['train', 'test', 'dev']:
        input_file = './data/' + settype + '.csv'
        dataset = read_csv(input_file)
        paris_label = read_pairs_label(input_file)
        get_ari_both_with_label(dataset, paris_label)
