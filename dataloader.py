"""File to load the datasets for each task"""

from datasets import load_dataset
from transformers import BertTokenizer

def load_and_process_dataset(dataset, encode_batch):
    # Encode the input data
    dataset = dataset.map(encode_batch, batched=True)
    # The transformers model expects the target class column to be named "labels"
    dataset = dataset.rename_column("label", "labels")
    # Transform to pytorch tensors and only output the required columns
    dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])

    id2label = {id: label for (id, label) in enumerate(dataset["train"].features["labels"].names)}
    return dataset, id2label

def load_cb():
    """Load glue dataset"""
    dataset = load_dataset("super_glue", "cb")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            max_length=180,
            truncation=True,
            padding="max_length"
        )
    return load_and_process_dataset(dataset, encode_batch), "validation"

def load_sst2():
    """Load glue dataset"""
    dataset = load_dataset("glue", "sst2")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        return tokenizer(
            batch['sentence'],
            max_length=180,
            truncation=True,
            padding="max_length"
        )
    return load_and_process_dataset(dataset, encode_batch), "validation"

def load_boolq():
    """Load glue dataset"""
    dataset = load_dataset("super_glue", "boolq")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        return tokenizer(
            batch["question"],
            batch["passage"],
            max_length=180,
            truncation=True,
            padding="max_length"
        )
    return load_and_process_dataset(dataset, encode_batch), "validation"

def load_mnli():
    """Load glue dataset"""
    dataset = load_dataset("glue", "mnli")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            max_length=180,
            truncation=True,
            padding="max_length"
        )

    return load_and_process_dataset(dataset, encode_batch), "validation_matched"

def load_qqp():
    """Load glue dataset"""
    dataset = load_dataset("glue", "qqp")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        return tokenizer(
            batch["question1"],
            batch["question2"],
            max_length=180,
            truncation=True,
            padding="max_length"
        )

    return load_and_process_dataset(dataset, encode_batch), "validation"

def load_wgrande():
    """Load glue dataset"""
    dataset = load_dataset("winogrande", "winogrande_xl")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # def encode_batch(batch):
    #     """Encodes a batch of input data using the model tokenizer."""
    #     return tokenizer(
    #         batch["question1"],
    #         batch["question2"],
    #         max_length=180,
    #         truncation=True,
    #         padding="max_length"
    #     )

    return load_and_process_dataset(dataset, encode_batch)


