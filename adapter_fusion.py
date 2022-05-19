"""Replicating table 1 from the AdapterFusion paper"""

from transformers import (
    BertTokenizer,
    EarlyStoppingCallback,
    BertConfig,
    BertModelWithHeads,
    TrainingArguments,
    AdapterTrainer,
    EvalPrediction,
    DataCollatorWithPadding
)
from transformers.adapters.composition import Fuse
import transformers
transformers.logging.set_verbosity_error()

import numpy as np
import torch

import argparse
from time import gmtime, strftime

import dataset_loader as dataset_loader


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
classification_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def multiple_choice_collator(features):
    labels = [feature.pop('labels') for feature in features]
    batch_size = len(features)
    num_choices = len(features[0]["input_ids"])
    flattened_features = [
        [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
    ]
    flattened_features = sum(flattened_features, [])

    batch = tokenizer.pad(
        flattened_features,
        return_tensors="pt",
    )

    batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
    batch["labels"] = torch.tensor(labels, dtype=torch.int64)
    return batch


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

def setup_adapter_fusion(model, id2label, task):
    # Load the pre-trained adapters we want to fuse
    model.load_adapter("nli/multinli@ukp", load_as="multinli", with_head=False)
    model.load_adapter("sts/qqp@ukp", load_as="qqp", with_head=False)
    model.load_adapter("sentiment/sst-2@ukp", load_as="sst", with_head=False)
    model.load_adapter("comsense/winogrande@ukp", load_as="wgrande", with_head=False)
    model.load_adapter("sentiment/imdb@ukp", load_as="imdb", with_head=False)
    model.load_adapter("comsense/hellaswag@ukp", load_as="hswag", with_head=False)
    model.load_adapter("comsense/siqa@ukp", load_as="siqa", with_head=False)
    model.load_adapter("comsense/cosmosqa@ukp", load_as="cqa", with_head=False)
    model.load_adapter("nli/scitail@ukp", load_as="scitail", with_head=False)
    model.load_adapter("argument/ukpsent@ukp", load_as="argument", with_head=False)
    model.load_adapter("comsense/csqa@ukp", load_as="csqa", with_head=False)
    model.load_adapter("qa/boolq@ukp", load_as="boolq", with_head=False)
    model.load_adapter("sts/mrpc@ukp", load_as="mrpc", with_head=False)
    model.load_adapter("nli/sick@ukp", load_as="sick", with_head=False)
    model.load_adapter("nli/rte@ukp", load_as="rte", with_head=False)
    model.load_adapter("nli/cb@ukp", load_as="cb", with_head=False)

    # Add a fusion layer for all loaded adapters
    adapter_setup = Fuse("multinli", "qqp", "sst", "wgrande", "imdb", "hswag",
                         "siqa", "cqa", "scitail", "argument", "csqa", "boolq",
                         "mrpc", "sick", "rte", "cb")
    model.add_adapter_fusion(adapter_setup)
    model.set_active_adapters(adapter_setup)

    # Add a classification head for our target task
    # Certain tasks requires multiple choice head instead of classification
    if task in ["hswag", "siqa", "cqa", "csqa"]:
        model.add_multiple_choice_head(task, num_choices=len(id2label))
    else:
        model.add_classification_head(task, num_labels=len(id2label))
    model.train_adapter_fusion(adapter_setup)
    return model

def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}

def train_model(model, training_args, dataset, collator_fn, args):
    """Train the model with training_args, with EarlyStopping enabled."""
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_accuracy,
        data_collator=collator_fn
    )
    callback = EarlyStoppingCallback(early_stopping_patience=2)
    trainer.add_callback(callback)

    with open(args.outfile, 'a') as outfile:
        outfile.write(f"Start training\t{strftime('%Y-%m-%d %H:%M:%S', gmtime())}\n")

    trainer.train()

    #When loading the model, it is put back on the CPU, so manually put it back
    #on the GPU for evaluation
    if torch.cuda.is_available():
        model = model.to('cuda')
    accuracy = trainer.evaluate()["eval_acc"]

    with open(args.outfile, 'a') as outfile:
        outfile.write(f"End training\t{strftime('%Y-%m-%d %H:%M:%S', gmtime())}\n")

    with open(args.outfile, 'a') as outfile:
        outfile.write(f"Evaluation accuracy: {accuracy}\n")

training_args = TrainingArguments(
    learning_rate=5e-5,
    num_train_epochs=10, #Most of the time only 1 or 2 epochs are done, so EarlyStopping is a must
    per_device_train_batch_size=2, #higher has memory problems on lisa, but definitely possible. Can be set as argument
    per_device_eval_batch_size=2,
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    output_dir="training_output",
    overwrite_output_dir=True,
    load_best_model_at_end=True, # Must be used for EarlyStopping
    remove_unused_columns=False, # Important to ensure the dataset labels are properly passed to the model
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', type=str,
                        help='file to write results to')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='per device batch size for training and eval')
    parser.add_argument('--tasks', nargs='+', default=["mnli", "qqp", "sst", "wgrande", "imdb", "hswag", "siqa", "cqa", "scitail", "argument", "csqa", "boolq", "mrpc", "sick", "rte", "cb"],
                        help='list of tasks, seperated by a space')
    args = parser.parse_args()

    training_args.per_device_train_batch_size = args.batch_size
    training_args.per_device_eval_batch_size = args.batch_size

    for task in args.tasks:
        with open(args.outfile, 'a') as outfile:
            outfile.write(f"[Training fusion ST-A for task {task}]\n")

        dataset, id2label = dataset_loader.load_dataset_by_name(task)

        model = load_bert_model(id2label)
        model = setup_adapter_fusion(model, id2label, task)

        # the multiple-choice tasks have a different data collator function
        if task in ["hswag", "siqa", "cqa", "csqa"]:
            collator_fn = multiple_choice_collator
        else:
            collator_fn = classification_collator

        train_model(model, training_args, dataset, collator_fn, args)
