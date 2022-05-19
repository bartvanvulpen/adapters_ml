import torch
from sampler import (
    TaskBatchSampler,
    split_batch,
)
import torch.utils.data as data
from dataset_loader import load_dataset_by_name, load_dataset_from_file


def combine_train_valid(name, LOADED_DATASETS):
    """
    Combine the train and validation sets into one big dataset
    """

    (ds, id2label) = LOADED_DATASETS[name]

    # handle validation keys
    key = 'validation'
    if name == 'imdb':
        key = 'test'

    if name == 'argument':
        all_inputs = torch.cat((torch.stack([x['input_ids'] for x in ds['train']]),
                                torch.stack([x['input_ids'] for x in ds['validation']])), dim=0)
        all_token_types = torch.cat((torch.stack([x['token_type_ids'] for x in ds['train']]),
                                torch.stack([x['token_type_ids'] for x in ds['validation']])), dim=0)
        all_masks = torch.cat((torch.stack([x['attention_mask'] for x in ds['train']]),
                                     torch.stack([x['attention_mask'] for x in ds['validation']])), dim=0)
        all_labels = torch.cat((torch.stack([x['labels'] for x in ds['train']]),
                              torch.stack([x['labels'] for x in ds['validation']])), dim=0)

    else:
        # combine input data from train and validation set
        all_inputs = torch.cat((torch.tensor(ds["train"]["input_ids"]), torch.tensor(ds[key]["input_ids"])), dim=0)
        all_token_types = torch.cat((torch.tensor(ds["train"]["token_type_ids"]), torch.tensor(ds[key]["token_type_ids"])), dim=0)
        all_masks = torch.cat((torch.tensor(ds["train"]["attention_mask"]), torch.tensor(ds[key]["attention_mask"])), dim=0)
        all_labels = torch.cat((torch.tensor(ds["train"]["labels"]), torch.tensor(ds[key]["labels"])), dim=0)

    return {
        "input_ids": all_inputs,
        "token_type_ids": all_token_types,
        "attention_mask": all_masks,
        "labels": all_labels
    }




class NLPDataset(data.Dataset):
    """
    Generic dataset to handle all the NLP datasets in a similar way
    """

    def __init__(self, tasks, input_ids, token_type_ids, attention_mask, labels):
        super().__init__()
        self.tasks = tasks
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __getitem__(self, idx):
        task = self.tasks[idx]
        x = (self.input_ids[idx], self.token_type_ids[idx], self.attention_mask[idx])
        label = self.labels[idx]
        return task, x, label

    def __len__(self):
        return self.labels.shape[0]


def dataset_from_tasks(dataset, tasks, **kwargs):
    """
    Create a new dataset based on filtering out specific tasks in the combined dataset
    """

    task_mask = (dataset["tasks"][:, None] == tasks[None, :]).any(dim=-1)

    dataset = NLPDataset(
        tasks=dataset["tasks"][task_mask],
        input_ids=dataset["input_ids"][task_mask],
        token_type_ids=dataset["token_type_ids"][task_mask],
        attention_mask=dataset["attention_mask"][task_mask],
        labels=dataset["labels"][task_mask],
        **kwargs
    )

    return dataset if max_num == None else dataset[:max_num]




def get_train_val_loaders(train_datasets, val_datasets, num_workers=0, K_SHOT = 4):

    print('Loading all datasets...')

    dataset_names = train_datasets + val_datasets

    LOADED_DATASETS = {name: load_dataset_from_file(name) for name in dataset_names}

    print('Done loading all datasets')

    print('Combining...')
    DATASETS = {ds: combine_train_valid(ds, LOADED_DATASETS) for ds in LOADED_DATASETS.keys()}
    TASK_IDS = {name: id for id, name in enumerate(DATASETS.keys())}

    combined_dataset = {
        "tasks": torch.hstack(
            [
                torch.tensor([TASK_IDS[ds_name]] * len(ds["labels"]))
                for ds_name, ds in DATASETS.items()
            ]
        ),
        "input_ids": torch.vstack([ds["input_ids"] for ds in DATASETS.values()]),
        "token_type_ids": torch.vstack([ds["token_type_ids"] for ds in DATASETS.values()]),
        "attention_mask": torch.vstack([ds["attention_mask"] for ds in DATASETS.values()]),
        "labels": torch.hstack([ds["labels"] for ds in DATASETS.values()]),
    }

    print('Creating datasets...')
    train_set = dataset_from_tasks(
        combined_dataset, torch.tensor([TASK_IDS[ds] for ds in train_datasets])
    )
    print('TRAIN SET SIZE:', len(train_set))
    val_set = dataset_from_tasks(
        combined_dataset, torch.tensor([TASK_IDS[ds] for ds in val_datasets])
    )

    print('VAL SET SIZE:', len(val_set))

    # imdb
    # hswag
    # mrpc
    # argument
    # scitail

    # Training set
    train_protomaml_sampler = TaskBatchSampler(
        train_set.tasks,
        train_set.labels,
        include_query=True,
        K_shot=K_SHOT,
        batch_size=16,
        shuffle=True,
    )

    print('Creating train dataloader...')
    train_protomaml_loader = data.DataLoader(
        train_set,
        batch_sampler=train_protomaml_sampler,
        collate_fn=train_protomaml_sampler.get_collate_fn(),
        num_workers=num_workers,
    )


    # Validation set
    val_protomaml_sampler = TaskBatchSampler(
        val_set.tasks,
        val_set.labels,
        include_query=True,
        K_shot=K_SHOT,
        batch_size=1,  # We do not update the parameters, hence the batch size is irrelevant here
        shuffle=False
    )
    print('Creating val dataloader...')
    val_protomaml_loader = data.DataLoader(
        val_set,
        batch_sampler=val_protomaml_sampler,
        collate_fn=val_protomaml_sampler.get_collate_fn(),
        num_workers=num_workers,
    )

    return train_protomaml_loader, val_protomaml_loader
