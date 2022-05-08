"""File to load the datasets for each task"""

from datasets import load_dataset
from transformers import BertTokenizer

def load_and_process_dataset(dataset, encode_batch, label_name, labels=None):
    # The transformers model expects the target class column to be named "labels"
    dataset = dataset.rename_column(label_name, "labels")
    # Encode the input data
    dataset = dataset.map(encode_batch, batched=True)

    # Transform to pytorch tensors and only output the required columns
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    if labels is None:
        id2label = {id: label for (id, label) in enumerate(dataset["train"].features["labels"].names)}
    else:
        id2label = {id: label for (id, label) in enumerate(labels)}

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
            max_length=180,
            truncation=True,
            padding="max_length",
            add_special_tokens=True
        )
        if not labels is None:
            tokens["labels"] = [label2id[label] if label in label2id else -1 for label in batch["labels"]]
        return tokens
    return load_and_process_dataset(dataset, encode_batch, label_name, labels=labels)

def load_specific_multi_choice_dataset(dataset_name, task_name, sentence_a_inputs, sentence_b_inputs, label_name, labels=None):
    """Load datasets with multiple options as inputs, e.g. quess the correct ending"""
    dataset = load_dataset(dataset_name, task_name)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if not labels is None:
        label2id = {label: id for (id, label) in enumerate(labels)}
    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer.
        If the labels are not given as ints, change them to ints"""
        len_batch = len(batch[sentence_a_inputs[0]])
        num_choice = len(sentence_b_inputs)

        all_encoded = {"input_ids": [], "attention_mask": []}
        batch_inputs_a = [batch[name] for name in sentence_a_inputs]
        batch_inputs_b = [batch[name] for name in sentence_b_inputs]
        for i in range(len_batch):
            sentences_a = [' '.join([column[i] for column in batch_inputs_a]) for _ in range(num_choice)]
            sentences_b = [column[i] for column in batch_inputs_b]
            encoded = tokenizer(
                sentences_a,
                sentences_b,
                max_length=180,
                truncation=True,
                padding="max_length",
            )
            all_encoded["input_ids"].append(encoded["input_ids"])
            all_encoded["attention_mask"].append(encoded["attention_mask"])
        if not labels is None:
            all_encoded["labels"] = [label2id[label] if label in label2id else -1 for label in batch["labels"]]
        else:
            all_encoded["labels"] = batch["labels"]
        return all_encoded
    return load_and_process_dataset(dataset, encode_batch, label_name, labels=labels)


def load_dataset_by_name(name):
    if name == "mnli":
        return load_specific_dataset("glue", "mnli", ["premise", "hypothesis"], "label")
    elif name == "qqp":
        return load_specific_dataset("glue", "qqp", ["question1", "question2"], "label")
    elif name == "sst":
        return load_specific_dataset("glue", "sst2", ["sentence"], "label")
    elif name == "wgrande":
        return load_specific_multi_choice_dataset("winogrande", "winogrande_xl", ["sentence"], ["option1", "option2"], "answer", labels=["1", "2"])
    elif name == "imdb":
        return load_specific_dataset("imdb", "plain_text", ["text"], "label")
    elif name == "hswag":
        raise NotImplementedError()
        #return load_specific_dataset("hellaswag", "default", ["activity_label", "ctx", "endings"], "label")
    elif name == "siqa":
        #TODO: test
        return load_specific_multi_choice_dataset("social_i_qa", "default", ["context", "question"], ["answerA", "answerB", "answerC"], "label")
    elif name == "cqa":
        #TODO: test
        return load_specific_multi_choice_dataset("cosmos_qa", "default", ["context", "question"], ["answer0", "answer1", "answer2", "answer3"], "label")
    elif name == "scitail":
        return load_specific_dataset("scitail", "tsv_format", ["premise", "hypothesis"], "label", labels=["neutral", "entails"])
    elif name == "argument":
        raise NotImplementedError() #I can't find this dataset
    elif name == "csqa":
        #TODO: test
        return load_specific_multi_choice_dataset("cosmos_qa", "default", ["context", "question"], ["answer0", "answer1", "answer2", "answer3"], "label")
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
