import logging
import pickle
import json
import os
import torch
from transformers import BertTokenizer, RobertaTokenizer
import torch.nn as nn


# Loading Pickle Format File
def load_pickle_data(path):
    return pickle.load(open(path, 'rb'))


# Loading Json Format File
def load_json_data(path):
    return json.load(open(path, 'r', encoding='utf-8'))


# Defining Logger for Experiment Records
def define_logger(hps):
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
    logging.basicConfig(format=formatter, level=logging.INFO)
    logger = logging.getLogger(hps.log_name)
    
    console_handler = logging.StreamHandler()
    console_handler.formatter = formatter
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    file_path = os.path.join(hps.log_dir, hps.log_name)
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


# Process Data
def process(hps, data):
    data = data['instances']
    chains = []
    chain_contexts = []
    labels = []
    threshold_labels = []
    scene_labels = []
    other_labels = []
    for instance in data:
        wrong_types = instance['wrong_type']
        chains.append(list(instance['events']))
        tmp_context = list(instance['short_contexts'])
        if len(tmp_context) != 4:
            tmp_context = list(instance['short_contexts'])
        else:
            tmp_context = [instance['short_contexts'][index] if 'NULL' in tmp_context[index] else tmp_context[index] for index in range(4)]
        chain_contexts.append(tmp_context)
        labels.append(5 if instance['label'] == 0 else instance['label'])
        threshold_labels.append(1 if 'threshold problem' in wrong_types else 0)
        scene_labels.append(1 if 'scene problem' in wrong_types else 0)
        other_labels.append(1 if 'other problem' in wrong_types else 0)
    return chains, chain_contexts, torch.LongTensor(labels), torch.LongTensor(threshold_labels), \
           torch.LongTensor(scene_labels), torch.LongTensor(other_labels)


# Tokenize Pad Data for Main Model
def tokenization(hps, chains, chain_contexts):
    tokenizer = BertTokenizer.from_pretrained(hps.transformer_dir)

    chains_ids, chains_mask = [], []
    for chain in chains:
        if hps.language == 'english':
            chain = [event.replace('_', ' ') for event in chain]
        chain_outputs = tokenizer(chain, padding=False)
        chain_ids, chain_mask = [], []
        for i in range(len(chain_outputs.input_ids)):
            chain_ids += chain_outputs.input_ids[i][1:]
            chain_mask += chain_outputs.attention_mask[i][1:]
        chain_ids = [tokenizer.cls_token_id] + chain_ids
        chain_mask = [1] + chain_mask
        chains_ids.append(chain_ids)
        chains_mask.append(chain_mask)

    contexts_ids, contexts_mask = [], []
    for context in chain_contexts:
        context_output = tokenizer(context, padding=False)
        context_ids, context_mask = [], []
        for i in range(len(context_output.input_ids)):
            context_ids += context_output.input_ids[i][1:]
            context_mask += context_output.attention_mask[i][1:]
        context_ids = [tokenizer.cls_token_id] + context_ids
        context_mask = [1] + context_mask
        contexts_ids.append(context_ids)
        contexts_mask.append(context_mask)

    chain_max_len = max([len(chain) for chain in chains_ids])
    chain_mask_max_len = max([len(chain) for chain in chains_mask])
    context_max_len = max([len(context) for context in contexts_ids])
    context_mask_max_len = max([len(context) for context in contexts_mask])

    chains_ids = [chain + [tokenizer.pad_token_id] * (chain_max_len - len(chain)) for chain in chains_ids]
    chains_mask = [chain + [0] * (chain_mask_max_len - len(chain)) for chain in chains_mask]
    contexts_ids = [context + [tokenizer.pad_token_id] * (context_max_len - len(context)) for context in contexts_ids]
    contexts_mask = [context + [0] * (context_mask_max_len - len(context)) for context in contexts_mask]

    print("context_max_len:{}".format(context_max_len))

    return torch.LongTensor(chains_ids), torch.LongTensor(chains_mask), \
           torch.LongTensor(contexts_ids), torch.LongTensor(contexts_mask)


# Evaluation Function
def evaluation(hps, model, dataloader, epslion):
    TP, FP, TN, FN = 0, 0, 0, 0
    three, four, five = 0, 0, 0
    three_count, four_count, five_count = 0, 0, 0
    exact_match = 0
    instance_count = 0
    softmax = nn.Softmax(-1)

    with torch.no_grad():
        for batch in dataloader:
            if hps.cuda:
                batch = tuple(term.cuda() for term in batch)
            
            if torch.is_tensor(epslion):
                epslion = epslion[:batch[0].size(0)]

            outputs = model(batch, epslion)
            logits = outputs[-3]
            porbs = softmax(logits)
            predict_labels = torch.argmax(porbs, -1)
            predictions = predict_labels.cpu().tolist()
            labels = batch[4].cpu().tolist()

            true_labels = []
            for label in labels:
                if label == 5:
                    temp = [1 for _ in range(3)]
                    three_count += 1
                    four_count += 1
                    five_count += 1
                else:
                    if label == 4:
                        three_count += 1
                        four_count += 1
                        five_count += 1
                    elif label == 3:
                        three_count += 1
                        four_count += 1
                    else:
                        three_count += 1
                    temp = [1 for _ in range(label - 2)]
                    temp.append(0)
                true_labels.append(temp)
                instance_count += len(temp)

            for i in range(len(true_labels)):
                # if true_labels[i] == predictions[i][:len(true_labels[i])]:
                #     exact_match += len(true_labels[i])

                for j in range(len(true_labels[i])):
                    if predictions[i][j] == true_labels[i][j] == 1:
                        exact_match += 1
                        TP += 1
                        if j == 0:
                            three += 1
                        elif j == 1:
                            four += 1
                        else:
                            five += 1
                    elif predictions[i][j] == true_labels[i][j] == 0:
                        exact_match += 1
                        TN += 1
                        if j == 0:
                            three += 1
                        elif j == 1:
                            four += 1
                        else:
                            five += 1
                    elif predictions[i][j] == 1 and true_labels[i][j] == 0:
                        FP += 1
                    else:
                        FN += 1

    Accuracy = exact_match / instance_count
    assert (TP + TN) / (TP + TN + FP + FN) == Accuracy
    Accuracy3 = three / three_count
    Accuracy4 = four / four_count
    Accuracy5 = five / five_count
    try:
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = 2 * Precision * Recall / (Precision + Recall)
    except:
        Precision, Recall, F1 = 0, 0, 0

    return Precision, Recall, F1, Accuracy, (Accuracy3, Accuracy4, Accuracy5)



