"""Replicating table 1 from the AdapterFusion paper"""

from datasets import load_dataset

import transformers
from transformers import BertTokenizer, EarlyStoppingCallback
from transformers import BertConfig, BertModelWithHeads
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from transformers.adapters.composition import Fuse
transformers.logging.set_verbosity_error()

import numpy as np
import torch
import argparse

import dataloader as dataloader

def load_bert_model(id2label):
    config = BertConfig.from_pretrained(
        "bert-base-uncased",
        id2label=id2label,
    )
    model = BertModelWithHeads.from_pretrained(
        "bert-base-uncased",
        config=config,
    )
    return model

def setup_adapter(model, id2label, task):
    if task == "mnli":
        model.load_adapter("nli/multinli@ukp", load_as="multinli")
    elif task == "qqp":
        model.load_adapter("sts/qqp@ukp", load_as="qqp")
    elif task == "sst":
        model.load_adapter("sentiment/sst-2@ukp", load_as="sst")
    elif task == "wgrande":
        model.load_adapter("comsense/winogrande@ukp", load_as="wgrande")
        model.add_classification_head("wgrande", num_labels=len(id2label))
    elif task == "imdb":
        model.load_adapter("sentiment/imdb@ukp", load_as="imdb")
        model.add_classification_head("imdb", num_labels=len(id2label))
    elif task == "hswag":
        model.load_adapter("comsense/hellaswag@ukp", load_as="hswag")
        model.add_multiple_choice_head("hswag", num_choices=len(id2label))
    elif task == "siqa":
        model.load_adapter("comsense/siqa@ukp", load_as="siqa")
        model.add_multiple_choice_head("siqa", num_choices=len(id2label))
    elif task == "cqa":
        model.load_adapter("comsense/cosmosqa@ukp", load_as="cqa")
        model.add_classification_head("cqa", num_labels=len(id2label))
    elif task == "scitail":
        model.load_adapter("nli/scitail@ukp", load_as="scitail")
        model.add_classification_head("scitail", num_labels=len(id2label))
    elif task == "argument":
        model.load_adapter("argument/ukpsent@ukp", load_as="argument")
    elif task == "csqa":
        model.load_adapter("comsense/csqa@ukp", load_as="csqa")
        model.add_classification_head("csqa", num_labels=len(id2label))
    elif task == "boolq":
        model.load_adapter("qa/boolq@ukp", load_as="boolq")
    elif task == "mrpc":
        model.load_adapter("sts/mrpc@ukp", load_as="mrpc")
    elif task == "sick":
        model.load_adapter("nli/sick@ukp", load_as="sick")
    elif task == "rte":
        model.load_adapter("nli/rte@ukp", load_as="rte")
    elif task == "cb":
        model.load_adapter("nli/cb@ukp", load_as="cb")
        model.add_classification_head("cb", num_labels=len(id2label))
    else:
        raise NotImplementedError()

    # Set active adapter
    model.set_active_adapters(task)
    model.train_adapter(task)
    return model

def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}

def evaluate_model(model, dataset):
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_accuracy,
    )
    with open('results.txt', 'a') as outfile:
        outfile.write('eval acc pre-training: ' + str(trainer.evaluate()["eval_acc"]) + '\n')
    trainer.train()
    with open('results.txt', 'a') as outfile:
        outfile.write('eval acc post-training: ' + str(trainer.evaluate()["eval_acc"]) + '\n')

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=100,
    output_dir="training_output",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)

if __name__ == '__main__':
    tasks = ["mrpc", "scitail", "qqp", "wgrande", "hswag", "imdb", "siqa", "cqa", "argument", "csgq", "boolq", "rte", "cb"]

    for task in tasks:
        with open('results.txt', 'a') as outfile:
            outfile.write(f"[Evaluating the adapters for task {task}]\n")
        #try
        dataset, id2label = dataloader.load_dataset_by_name(task)
        #except Exception as e:
        #    with open('results.txt', 'a') as outfile:
        #        outfile.write(f"Couldn't load dataset because of exception: \n")
        #        outfile.write(str(e) + '\n')
        #    continue

        try:
            model = load_bert_model(id2label)
            model = setup_adapter(model, id2label, task)
        except Exception as e:
            with open('results.txt', 'a') as outfile:
                outfile.write(f"Couldn't load adapter because of exception: \n")
                outfile.write(str(e) + '\n')
            continue
        evaluate_model(model, dataset)
