from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
import json,evaluate, random, torch
import numpy as np

SEED = 7

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def load_dataset_no_test(ds):
    data = {'train': {}, 'dev': {}}

    data['train']['label'] = []
    data['train']['text'] = []
    data['dev']['label'] = []
    data['dev']['text'] = []

    with open(ds) as filehandle:
        json_data = json.load(filehandle)
    print('File ' +ds+ ' loaded.')

    for sample in json_data['train']:
        data['train']['text'].append(sample[0])
        data['train']['label'].append(sample[1])

    for sample in json_data['dev']:
        data['dev']['text'].append(sample[0])
        data['dev']['label'].append(sample[1])

    final_data = DatasetDict()
    for k, v in data.items():
        final_data[k] = Dataset.from_dict(v)

    return final_data


def load_dataset_no_dev(ds):
    data = {'train': {}, 'test': {}}

    data['train']['label'] = []
    data['train']['text'] = []
    data['test']['label'] = []
    data['test']['text'] = []

    with open(ds) as filehandle:
        json_data = json.load(filehandle)
    print('File ' +ds+ ' loaded.')

    for sample in json_data['train']:
        data['train']['text'].append(sample[0])
        data['train']['label'].append(sample[1])

    for sample in json_data['test']:
        data['test']['text'].append(sample[0])
        data['test']['label'].append(sample[1])

    final_data = DatasetDict()
    for k, v in data.items():
        final_data[k] = Dataset.from_dict(v)

    return final_data


def load_dataset(ds):
    data = {'train': {}, 'dev': {}, 'test': {}}

    data['train']['label'] = []
    data['train']['text'] = []
    data['dev']['label'] = []
    data['dev']['text'] = []
    data['test']['label'] = []
    data['test']['text'] = []

    with open(ds) as filehandle:
        json_data = json.load(filehandle)
    print('File ' +ds+ ' loaded.')

    for sample in json_data['train']:
        data['train']['text'].append(sample[0])
        data['train']['label'].append(sample[1])

    for sample in json_data['dev']:
        data['dev']['text'].append(sample[0])
        data['dev']['label'].append(sample[1])

    for sample in json_data['test']:
        data['test']['text'].append(sample[0])
        data['test']['label'].append(sample[1])

    final_data = DatasetDict()
    for k, v in data.items():
        final_data[k] = Dataset.from_dict(v)

    return final_data


def tokenize_sequence(samples):
    if 'text2' in samples.keys():
        return tknz(samples["text"], samples["text2"], padding="max_length", truncation=True)
    else:
        return tknz(samples["text"], padding=True, truncation=True)


def load_model(n_lb):
    tokenizer_hf = AutoTokenizer.from_pretrained('roberta-large')
    model = AutoModelForSequenceClassification.from_pretrained('roberta-large', num_labels=n_lb, ignore_mismatched_sizes=True)

    return tokenizer_hf, model


def load_scheme_model(n_lb, path):
    tokenizer_hf = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=n_lb, ignore_mismatched_sizes=True)

    return tokenizer_hf, model


def load_local_model(path):
    tokenizer_hf = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)

    return tokenizer_hf, model


def compute_metrics(eval_preds):
    metric = evaluate.load("f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='macro')


def train_model(mdl, tknz, data):

    training_args = TrainingArguments(
        output_dir="models",
        evaluation_strategy="epoch",
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        learning_rate=1e-5,
        weight_decay=0.01,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=20,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        fp16=True
    )

    trainer = Trainer(
        model=mdl,
        args=training_args,
        train_dataset=data['train'],
        eval_dataset=data['test'],
        tokenizer=tknz,
        data_collator=DataCollatorWithPadding(tokenizer=tknz),
        compute_metrics=compute_metrics
    )

    trainer.train()

    return trainer


if __name__ == "__main__":

    PRETRAIN = True

    # num_labels = 24  # schemes
    num_labels = 8  # families

    # ds = 'data/eng/experiment_nlas_eng.json'
    # ds = 'data/eng/experiment_nlas_eng_grouped.json'
    # ds = 'data/eng/experiment_nlas_extended.json' # no-test
    # ds = 'data/eng/experiment_nlas_extended_grouped.json'# no-test
    # ds = 'data/qt-schemes/qt-schemes-nodef.json' # no-dev
    # ds = 'data/qt-schemes/qt-schemes-family-nodef.json' # no-dev
    # ds = 'data/qt-schemes/qt-schemes.json'  # qt-schemes
    ds = 'data/qt-schemes/qt-schemes-families.json'  # qt-schemes-families

    # LOAD DATA FOR THE MODE _no_dev or _no_test
    dataset = load_dataset_no_dev(ds)

    if PRETRAIN:

        # LOAD PRE_TRAINED ROBERTA
        # tknz, mdl = load_model(num_labels)

        # SchemeClassifier-ENG - AS-comp
        # SchemeClassifier2-ENG - AS-proc

        # tknz, mdl = load_scheme_model(num_labels, 'raruidol/SchemeClassifier2-ENG')

        tknz, mdl = load_local_model('models/af-proc')

        # TOKENIZE THE DATA
        tokenized_data = dataset.map(tokenize_sequence, batched=True)

        # TRAIN THE MODEL
        trainer = train_model(mdl, tknz, tokenized_data)

    else:

        # LOAD THE MODEL
        path_model = 'models/af-proc-dial'
        tknz, mdl = load_local_model(path_model)
        # tknz, mdl = load_scheme_model(num_labels, "raruidol/SchemeClassifier2-ENG")

        # TOKENIZE THE DATA
        shuffled_dataset = dataset.shuffle(seed=42)
        tokenized_data = shuffled_dataset.map(tokenize_sequence, batched=True)

        # INSTANTIATE THE MODEL
        trainer = Trainer(mdl)

    # GENERATE PREDICTIONS FOR DEV AND TEST
    try:
        dev_predictions = trainer.predict(tokenized_data['dev'])
        dev_predict = np.argmax(dev_predictions.predictions, axis=-1)
        mf1_dev = precision_recall_fscore_support(tokenized_data['dev']['label'], dev_predict, average='macro')
        print('Precision/Recall/F1/Support score in', ds, 'setup, DEV:', mf1_dev)
        print('Confusion matrix', ds)
        print(confusion_matrix(tokenized_data['dev']['label'], dev_predict))
    except:
        print('no dev')

    try:

        test_predictions = trainer.predict(tokenized_data['test'])
        test_predict = np.argmax(test_predictions.predictions, axis=-1)
        mf1_test = precision_recall_fscore_support(tokenized_data['test']['label'], test_predict, average='macro')

        print('Precision/Recall/F1/Support score in', ds, 'setup, TEST:', mf1_test)
        print('Confusion matrix',  ds)
        print(confusion_matrix(tokenized_data['test']['label'], test_predict))

    except:
        print('no test')
