import itertools
from datasets import Dataset, DatasetDict
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import json
import os
from torch.nn import functional as F
import torch
import pandas as pd

'''
MAP = {-1: "Default Inference", 0: "Position to Know", 1: "Expert Opinion", 2: "Direct ad Hominem",
                 3: "Inconsistent Commitment",
                 4: "Popular Practice", 5: "Popular Opinion", 6: "Analogy", 7: "Precedent", 8: "Example",
                 9: "Established Rule", 10: "Cause to Effect", 11: "Verbal Classification", 12: "Slippery Slope",
                 13: "Sign", 14: "Ignorance", 15: "Threat", 16: "Waste", 17: "Sunk Costs", 18: "Witness Testimony",
                 19: "Best Explanation"}
'''

MAP = {-1: "Default Inference", 0: "Position to Know Argument", 1: "Ad Hominem Argument", 2: "Popular Acceptance",
       3: "Defeasible Rule-based Argument", 4: "Argument Based on Cases", 5: "Chained Argument from Rules and Cases",
       6: "Discovery Argument", 7: "Practical Reasoning"}


TOKENIZER = AutoTokenizer.from_pretrained("raruidol/SchemeClassifier3-ENG")
MODEL = AutoModelForSequenceClassification.from_pretrained("raruidol/SchemeClassifier3-ENG")


def preprocess_data(filexaif):
    inferences_id = []
    data = {'text': []}

    for node in filexaif['nodes']:
        if node['type'] == 'RA':
            # Uncomment for classifying the unspecified inferences only
            # if node['text'] == 'Default Inference':
            id = node['nodeID']
            inferences_id.append(id)
            claim = ''
            premise = ''
            for edge in filexaif['edges']:
                if edge['fromID'] == id:
                    for node in filexaif['nodes']:
                        if node['nodeID'] == edge['toID'] and node['type'] == 'I':
                            claim += node['text']+'. '
                elif edge['toID'] == id:
                    for node in filexaif['nodes']:
                        if node['nodeID'] == edge['fromID'] and node['type'] == 'I':
                            premise += node['text']+'. '
            data['text'].append(premise+claim)

    final_data = Dataset.from_dict(data)

    return final_data, inferences_id


def tokenize_sequence(samples):
    return TOKENIZER(samples["text"], padding=True, truncation=True)


def make_predictions(trainer, tknz_data):
    predicted_logprobs = trainer.predict(tknz_data)
    labels = []
    for sample in predicted_logprobs.predictions:
        torch_logits = torch.from_numpy(sample)
        probabilities = F.softmax(torch_logits, dim=-1).numpy()
        valid_check = probabilities > 0.95
        if True in valid_check:
            labels.append(np.argmax(sample, axis=-1))
        else:
            labels.append(-1)

    return labels


def output_xaif(idents, labels, fileaif):
    predicts = {}
    if isinstance(labels, np.ndarray) or isinstance(labels, list):
        for i in range(len(labels)):
            lb = MAP[labels[i]]
            if lb in predicts:
                predicts[lb] += 1
            else:
                predicts[lb] = 1
            id = idents[i]
            for node in fileaif['nodes']:
                if node['nodeID'] == id:
                    node['scheme'] = lb
                    node['text'] = lb
                    break

    return fileaif, predicts


def scheme_classification(xaif, predicted_d):

    # Generate a HF Dataset from all the "I" node pairs to make predictions from the xAIF file
    # and a list of tuples with the corresponding "I" node ids to generate the final xaif file.
    dataset, ids = preprocess_data(xaif)

    for arg in dataset['text']:
        predicted_d['text'].append(arg)

    # Tokenize the Dataset.
    tokenized_data = dataset.map(tokenize_sequence, batched=True)
    if len(tokenized_data['text']) == 0:
        return xaif, 0, {}, predicted_d

    # Instantiate HF Trainer for predicting.
    trainer = Trainer(MODEL)

    # Predict the list of labels for all the pairs of "I" nodes.
    labels = make_predictions(trainer, tokenized_data)

    for lb in labels:
        predicted_d['scheme'].append(MAP[lb])

    # Prepare the xAIF output file.
    out_xaif, iter_predicts = output_xaif(ids, labels, xaif)

    return out_xaif, len(labels), iter_predicts, predicted_d


if __name__ == "__main__":
    path = 'cutiestestrun18march2021/'  # -> Total Inferences: 149
    # Predictions:  {'Discovery Argument': 27, 'Defeasible Rule-based Argument': 12, 'Practical Reasoning': 11,
    # 'Default Inference': 38, 'Position to Know Argument': 42, 'Ad Hominem Argument': 16,
    # 'Argument Based on Cases': 2, 'Popular Acceptance': 1}

    # path = 'qt02092021wtxt/'  # -> Total Inferences:  171
    # Predictions:  {'Discovery Argument': 34, 'Defeasible Rule-based Argument': 7, 'Default Inference': 46,
    # 'Argument Based on Cases': 1, 'Practical Reasoning': 35, 'Ad Hominem Argument': 26,
    # 'Position to Know Argument': 22}

    # path = 'qt19082021wtxt/'  # -> Total Inferences:  161
    # Predictions:  {'Chained Argument from Rules and Cases': 1, 'Defeasible Rule-based Argument': 7,
    # 'Default Inference': 40, 'Discovery Argument': 27, 'Argument Based on Cases': 1, 'Position to Know Argument': 43,
    # 'Ad Hominem Argument': 28, 'Practical Reasoning': 14}

    n_inferences = 0
    total_predicts = {}
    predicted_dict = {'text': [], 'scheme': []}
    for filename in os.listdir(path):
        if filename.split('.')[1] == 'json' and 'out' not in filename:
            print(filename)
            with open(path + filename) as filehandle:
                content = json.load(filehandle)

            out, infer, iter_predicts, predicted_dict = scheme_classification(content, predicted_dict)
            n_inferences += infer
            print(iter_predicts)
            total_predicts = {k: total_predicts.get(k, 0) + iter_predicts.get(k, 0) for k in set(total_predicts) | set(iter_predicts)}
            print('Inferences: ', infer)

            #with open(path+'out_'+filename, "w") as outfile:
            #    json.dump(out, outfile, indent=4)

    predicted_df = pd.DataFrame.from_dict(predicted_dict)
    #predicted_df.to_csv(path+path.split('/')[0]+'.csv', index=False)
    print('Total Inferences: ', n_inferences)
    print('Predictions: ', total_predicts)


