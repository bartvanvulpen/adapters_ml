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

def setup_adapter_fusion(model, id2label):
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
    model.add_classification_head("cb", num_labels=len(id2label))
    model.train_adapter_fusion(adapter_setup)
    return model

def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}

def train_model(model, training_args, dataset, args):
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_accuracy,
    )
    trainer.train()
    trainer.evaluate()

    print('=================DONE TRAINING============')
    total = 0
    total_correct = 0
    for batch in dataset["test"]:
        print(batch)
        labels = batch["labels"]
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        print(output.logits.cpu())
        print(torch.argmax(output.logits.cpu()), labels)
        print(torch.argmax(output.logits.cpu()) == labels, torch.sum(torch.argmax(output.logits.cpu()) == labels))
        total_correct += torch.sum(torch.argmax(output.logits.cpu()) == labels)
        total += 1
        print(total, total_correct, total_correct / total)

    #predictions = trainer.predict(test_dataset=dataset["test"]).predictions
    #test_results = compute_accuracy(predictions)
    #output_test_file = os.path.join(training_args.output_dir, f"test_results.txt")

    #with open(output_test_file, "w") as writer:
    #    writer.write(f"accuracy for {args.task} task: {test_results['acc']}")

training_args = TrainingArguments(
    learning_rate=5e-5,
    num_train_epochs=1,
    per_device_train_batch_size=8, #higher has memory problems on lisa
    per_device_eval_batch_size=8,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    lr_scheduler_type="constant",
    output_dir="training_output",
    overwrite_output_dir=True,
    do_predict=True,
    #load_best_model_at_end=True, <- this does not work currently, probably a bug from AdapterHub
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='cb',
                        help='task to train the ST-A fusion for')

    args = parser.parse_args()

    dataset, id2label = dataloader.load_dataset_by_name(args.task)

    model = load_bert_model(id2label)
    model = setup_adapter_fusion(model, id2label)
    train_model(model, training_args, dataset, args)
