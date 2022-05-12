"""File to load the datasets for each task"""

from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import Dataset
import csv
import os
import transformers
transformers.logging.set_verbosity_error()
import torch


def load_and_process_dataset(dataset, encode_batch, label_name, label2id=None, labels=None):
    # The transformers model expects the target class column to be named "labels"
    dataset = dataset.rename_column(label_name, "labels")
    # Encode the input data
    dataset = dataset.map(encode_batch, batched=True)

    # Transform to pytorch tensors and only output the required columns
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])

    if labels is None:
        id2label = {id: label for (id, label) in enumerate(dataset["train"].features["labels"].names)}
    else:
        id2label = {id: label for (id, label) in enumerate(labels)}

    print('num batches and size of batch of input_ids:', len(dataset["train"]["input_ids"]), len(dataset["train"]["input_ids"][0]))
    print('num batches and size of batch of attention_mask:', len(dataset["train"]["attention_mask"]), len(dataset["train"]["attention_mask"][0]))
    print('num batches and size of batch of labels:', len(dataset["train"]["labels"]))
    return dataset, id2label

def load_specific_dataset(dataset_name, task_name, inputs, label_name, labels=None):
    dataset = load_dataset(dataset_name, task_name)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if not labels is None:
        label2id = {label: id for (id, label) in enumerate(labels)}

    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer.
        If the labels are not given as ints, change them to ints"""
        batch_inputs = [batch[name] for name in inputs]
        tokens = tokenizer(
            *batch_inputs,
            truncation=True,
            padding="max_length",
            add_special_tokens=True
        )
        if not labels is None:
            tokens["labels"] = [label2id[label] if label in label2id else 0 for label in batch["labels"]]
        return tokens
    return load_and_process_dataset(dataset, encode_batch, label_name, labels=labels)

def load_specific_dataset_with_encoder(dataset_name, task_name, encode_batch, label_name, labels=None):
    """Load datasets with multiple options as inputs, e.g. quess the correct ending"""

def load_dataset_by_name(name):
    if name == "mnli":
        return load_specific_dataset("glue", "mnli", ["premise", "hypothesis"], "label")
    elif name == "qqp":
        return load_specific_dataset("glue", "qqp", ["question1", "question2"], "label")
    elif name == "sst":
        return load_specific_dataset("glue", "sst2", ["sentence"], "label")
    elif name == "wgrande":
        dataset = load_dataset("winogrande", "winogrande_xl")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        label2id = {"1": 0, "2": 1, "": 0}
        def encode_batch(examples):
            """Encodes a batch of input data using the model tokenizer."""
            all_encoded = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
            # Iterate through all examples in this batch
            for sentence, option1, option2 in zip(examples["sentence"], examples["option1"], examples["option2"]):
                sentence_a = sentence.replace('_', option1)
                sentence_b = sentence.replace('_', option2)
                encoded = tokenizer(
                    sentence_a,
                    sentence_b,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=True
                )
                all_encoded["input_ids"].append(encoded["input_ids"])
                all_encoded["attention_mask"].append(encoded["attention_mask"])
                all_encoded["token_type_ids"].append(encoded["token_type_ids"])
            all_encoded["labels"] = [label2id[label] if label in label2id else 0 for label in examples["labels"]]
            return all_encoded
        return load_and_process_dataset(dataset, encode_batch, "answer", labels=["1", "2"])
    elif name == "imdb":
        return load_specific_dataset("imdb", "plain_text", ["text"], "label")
    elif name == "hswag":
        dataset = load_dataset("hellaswag")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        def encode_batch(batch):
            """Encodes a batch of input data using the model tokenizer."""

            all_encoded = {"input_ids": [], "attention_mask": [], "labels": [], "token_type_ids": []}

            # Iterate through all examples in this batch
            for context, answers, label in zip(batch["ctx"], batch["endings"], batch["labels"]):
                # answers is an array or sentences, which is the format needed for the multiple-choice prediction head, hence we pass it as the second argument to tokenizer

                encoded = tokenizer(
                    [context for _ in range(4)],
                    answers,
                    truncation=True,
                    padding="max_length",
                )

                all_encoded["input_ids"].append(encoded["input_ids"])
                all_encoded["attention_mask"].append(encoded["attention_mask"])
                all_encoded["token_type_ids"].append(encoded["token_type_ids"])
                all_encoded["labels"].append(0 if label == "" else int(label))

            return all_encoded

        return load_and_process_dataset(dataset, encode_batch, "label", labels=["0", "1", "2", "3"])
    elif name == "siqa":
        dataset = load_dataset("social_i_qa", "default")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        label2id = {"1": 0, "2": 1, "3": 2, "": 0}
        def encode_batch(examples):
            """Encodes a batch of input data using the model tokenizer."""
            all_encoded = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
            # Iterate through all examples in this batch
            for context, question, answerA, answerB, answerC in zip(examples["context"], examples["question"], examples["answerA"], examples["answerB"], examples["answerC"]):
                sentences_a = [context + " " + question for _ in range(3)]
                sentences_b = [answerA, answerB, answerC]
                encoded = tokenizer(
                    sentences_a,
                    sentences_b,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=True
                )
                all_encoded["input_ids"].append(encoded["input_ids"])
                all_encoded["attention_mask"].append(encoded["attention_mask"])
                all_encoded["token_type_ids"].append(encoded["token_type_ids"])
            all_encoded["labels"] = [label2id[label] if label in label2id else 0 for label in examples["labels"]]
            return all_encoded
        return load_and_process_dataset(dataset, encode_batch, "label", labels=["1", "2", "3"])
    elif name == "cqa":
        dataset = load_dataset("cosmos_qa", "default")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        def encode_batch(examples):
            """Encodes a batch of input data using the model tokenizer."""
            all_encoded = {"input_ids": [], "attention_mask": []}
            # Iterate through all examples in this batch
            for context, question, answer0, answer1, answer2, answer3 in zip(examples["context"], examples["question"], examples["answer0"], examples["answer1"], examples["answer2"], examples["answer3"]):
                sentences_a = [context + " " + question for _ in range(4)]
                sentences_b = [answer0, answer1, answer2, answer3]
                encoded = tokenizer(
                    sentences_a,
                    sentences_b,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=True
                )
                all_encoded["input_ids"].append(encoded["input_ids"])
                all_encoded["attention_mask"].append(encoded["attention_mask"])
            return all_encoded
        return load_and_process_dataset(dataset, encode_batch, "label", labels=[0,1,2,3])
    elif name == "scitail":
        return load_specific_dataset("scitail", "tsv_format", ["premise", "hypothesis"], "label", labels=["neutral", "entails"])
    elif name == "argument":

        class ArgumentDatasetSplit(Dataset):
            def __init__(self):
                super().__init__()

                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.label2id = {'NoArgument': 0, 'Argument_against': 1, 'Argument_for': 2}
                self.data = []

            def add_item(self, topic, sentence, annotation):
                """
                adds an entry to the dataset
                """

                tokenized = self.tokenizer(
                    topic,
                    sentence,
                    truncation=True,
                    padding='max_length'
                )

                encoded_item = {'labels': torch.tensor(self.label2id[annotation])}
                encoded_item['input_ids'] = torch.tensor(tokenized['input_ids'])
                encoded_item['attention_mask'] = torch.tensor(tokenized['attention_mask'])
                encoded_item['token_type_ids'] = torch.tensor(tokenized['token_type_ids'])

                self.data.append(encoded_item)


            def __getitem__(self, index):
                return self.data[index]

            def __len__(self):
                return len(self.data)


        dataset_dict = {
            'train': ArgumentDatasetSplit(),
            'validation': ArgumentDatasetSplit(),
            'test': ArgumentDatasetSplit()
        }

        # loop over all files in the directory 'argument_dataset', containing the dataset files
        for filename in os.listdir('argument_dataset'):
            # get all .tsv files
            if filename.endswith('.tsv'):
                with open('argument_dataset/' + filename, newline='') as f:
                    # read each line from the tab-separated file
                    reader = csv.DictReader(f, delimiter='\t', quotechar='|')
                    for entry in reader:
                        # add each entry to the dataset
                        split = entry['set'] if entry['set'] != 'val' else 'validation'

                        dataset_dict[split].add_item(
                            entry['topic'],
                            entry['sentence'],
                            entry['annotation']
                        )

        id2label = {0: 'NoArgument', 1: 'Argument_against', 2: 'Argument_for'}

        return dataset_dict, id2label


    elif name == "csqa":
        dataset = load_dataset("commonsense_qa")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        label2id = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

        def encode_batch(batch):
            """Encodes a batch of input data using the model tokenizer."""
            all_encoded = {"input_ids": [], "attention_mask": [], "labels": []}
            # Iterate through all examples in this batch
            for question, choices, label in zip(batch["question"], batch["choices"], batch["labels"]):
                #choices is a json object containing labels and text.
                #The labels are always the same, so we can just take the text.
                encoded = tokenizer(
                    [question for _ in range(5)],
                    choices["text"],
                    truncation=True,
                    padding="max_length",
                )

                all_encoded["input_ids"].append(encoded["input_ids"])
                all_encoded["attention_mask"].append(encoded["attention_mask"])
                all_encoded["labels"].append(0 if label == "" else label2id[label])

            return all_encoded

        return load_and_process_dataset(dataset, encode_batch, "answerKey", labels=["A", "B", "C", "D", "E"])
    elif name == "boolq":
        return load_specific_dataset("boolq", "default", ["question", "passage"], "answer", labels=["true", "false"])
    elif name == "mrpc":
        return load_specific_dataset("glue", "mrpc", ["sentence1", "sentence2"], "label")
    elif name == "sick":
        return load_specific_dataset("sick", "default", ["sentence_A", "sentence_B", "en"], "label")
    elif name == "rte":
        return load_specific_dataset("glue", "rte", ["sentence1", "sentence2"], "label")
    elif name == "cb":
        return load_specific_dataset("super_glue", "cb", ["premise", "hypothesis"], "label")
    else:
        raise NotImplementedError()
