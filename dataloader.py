"""File to load the datasets for each task"""

from datasets import load_dataset
from transformers import BertTokenizer

def load_and_process_dataset(dataset, encode_batch, label_name):
    # Encode the input data
    dataset = dataset.map(encode_batch, batched=True)
    # The transformers model expects the target class column to be named "labels"
    dataset = dataset.rename_column(label_name, "labels")
    # Transform to pytorch tensors and only output the required columns
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    id2label = {id: label for (id, label) in enumerate(dataset["train"].features["labels"].names)}
    return dataset, id2label

def load_specific_dataset(dataset_name, task_name, inputs, label_name):
    dataset = load_dataset(dataset_name, task_name)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def encode_batch(batch):
        batch_inputs = [batch[name] for name in inputs]
        """Encodes a batch of input data using the model tokenizer."""
        return tokenizer(
            *batch_inputs,
            max_length=180,
            truncation=True,
            padding="max_length"
        )
    return load_and_process_dataset(dataset, encode_batch, label_name)

def load_dataset_by_name(name):
    if name == "mnli":
        return load_specific_dataset("glue", "mnli", ["premise", "hypothesis"], "label")
    elif name == "qqp":
        return load_specific_dataset("glue", "qqp", ["question1", "question2"], "label")
    elif name == "sst":
        return load_specific_dataset("glue", "sst2", ["sentence"], "label")
    elif name == "wgrande":
        return load_specific_dataset("winogrande", "winogrande_xl", ["sentence", "option1", "option2"], "answer")
    elif name == "imdb":
        return load_specific_dataset("imdb", "plain_text", ["text"], "label")
    elif name == "hswag":
        return load_specific_dataset("hellaswag", "default", ["activity_label", "ctx", "ctx_a", "ctx_b", "endings"], "label")
    elif name == "siqa":
        return load_specific_dataset("social_i_qa", "default", ["context", "question", "answerA", "answerB", "answerC"], "label")
    elif name == "cqa":
        return load_specific_dataset("cosmos_qa", "default", ["id", "context", "question", "answer0", "answer1", "answer2", "answer3"], "label")
    elif name == "scitail":
        return load_specific_dataset("scitail", "snli_format", ["premise", "hypothesis"], "label")
    elif name == "argument":
        raise NotImplementedError() #I can't find this dataset
    elif name == "csqa":
        raise NotImplementedError() #This dataset requires a bit more work, will do it later
    elif name == "boolq":
        return load_specific_dataset("boolq", "default", ["question", "passage"], "answer")
    elif name == "mrpc":
        return load_specific_dataset("glue", "mrpc", ["sentence1", "sentence2"], "label")
    elif name == "sick":
        raise NotImplementedError() #This dataset requires a bit more work as well, will do it later
    elif name == "rte":
        return load_specific_dataset("glue", "rte", ["sentence1", "sentence2"], "label")
    elif name == "cb":
        return load_specific_dataset("super_glue", "cb", ["premise", "hypothesis"], "label")
    else:
        raise NotImplementedError()
