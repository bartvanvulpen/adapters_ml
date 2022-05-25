import torch
from sampler import (
    TaskBatchSampler,
    split_batch,
    FewShotBatchSampler
)
import torch.utils.data as data
from dataset_loader import load_dataset_by_name, load_dataset_from_file


def combine_train_valid(name, LOADED_DATASETS, meta_test=False):
    """
    Combine the train and validation sets into one big dataset
    """

    (ds, id2label) = LOADED_DATASETS[name]


    # combine val and train set when training
    if meta_test == False:

        key = 'validation'
        if name == 'imdb':
            key = 'test'

        if name == 'mnli':
            key = 'validation_matched'


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

    # take validation set only when meta testing
    # there is no overlap, because meta testing task is unseen
    if meta_test == True:

            # take validation set only when testing
            key = 'validation'
            if name == 'imdb':
                key = 'test'

            if name == 'mnli':
                key = 'validation_matched'

            if name == 'argument':
                all_inputs = torch.stack([x['input_ids'] for x in ds['validation']])
                all_token_types = torch.stack([x['token_type_ids'] for x in ds['validation']])
                all_masks = torch.stack([x['attention_mask'] for x in ds['validation']])
                all_labels = torch.stack([x['labels'] for x in ds['validation']])

            else:
                # combine input data from train and validation set
                all_inputs = torch.tensor(ds[key]["input_ids"])
                all_token_types = torch.tensor(ds[key]["token_type_ids"])
                all_masks = torch.tensor(ds[key]["attention_mask"])
                all_labels = torch.tensor(ds[key]["labels"])

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

    return dataset

def get_num_classes(task_name):
    x = load_dataset_from_file(task_name)
    return len(x[1].keys())
    
def get_train_loader(train_datasets=[], K_SHOT=4, task_batch_size=16, num_workers=0):

   if train_datasets == []:
        raise RuntimeError('Train datasets must be specified!')

   if "hswag" in train_datasets or "siqa" in train_datasets or "cqa" in train_datasets or "csqa" in train_datasets:
        raise RuntimeError("Multiple choice datasets may not be used!")

   else:
        print('Loading training data...')
        LOADED_DATASETS = {name: load_dataset_from_file(name) for name in train_datasets}
        DATASETS = {ds: combine_train_valid(ds, LOADED_DATASETS, meta_test=False) for ds in LOADED_DATASETS.keys()}
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

        print(len(combined_dataset['input_ids']))
        train_set = dataset_from_tasks(
            combined_dataset, torch.tensor([TASK_IDS[ds] for ds in train_datasets])
        )

        # Training set
        train_protomaml_sampler = TaskBatchSampler(
            train_set.tasks,
            train_set.labels,
            include_query=True,
            K_shot=K_SHOT,
            batch_size=task_batch_size,
            shuffle=True,
        )

        train_protomaml_loader = data.DataLoader(
            train_set,
            batch_sampler=train_protomaml_sampler,
            collate_fn=train_protomaml_sampler.get_collate_fn(),
            num_workers=num_workers,
        )

        print('Done! TRAIN SET SIZE:', len(train_set))
        return train_protomaml_loader

def get_test_loaders(test_dataset, K_SHOT=4, full_dl_batch_size=8, num_workers=0):

    print('META-TEST DATASET:', test_dataset)
    test_dataset = [test_dataset]

    if test_dataset[0] in ["hswag", "siqa", "cqa", "csqa"]:
        raise RuntimeError("Multiple choice datasets may not be used!")

    else:
        print('Loading test data...')
        LOADED_DATASETS = {name: load_dataset_from_file(name) for name in test_dataset}
        DATASETS = {ds: combine_train_valid(ds, LOADED_DATASETS, meta_test=True) for ds in LOADED_DATASETS.keys()}
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

        test_set = dataset_from_tasks(
           combined_dataset, torch.tensor([TASK_IDS[ds] for ds in test_dataset])
        )

        # check labels correctness
        assert sum(test_set.labels) > 0, "Sum of labels can't be negative or zero, which means there are no labels for this data-split."

        # test sampler
        test_protomaml_sampler = FewShotBatchSampler(
           test_set.tasks,
           test_set.labels,
           include_query=False,
           K_shot=K_SHOT,
           shuffle=False,
        )

        full_test_loader = data.DataLoader(
            test_set,
            batch_size=full_dl_batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        sample_test_loader = data.DataLoader(
           test_set,
           batch_sampler=test_protomaml_sampler,
           num_workers=num_workers,
           shuffle=False
        )

        # Weird, if collate is added to dataloaders, weird stuff happening ??
        # so I removed and everything works like a charm :)


        print('Done! META-TEST SET SIZE:', len(test_set))

        return full_test_loader, sample_test_loader, test_protomaml_sampler

    # # Validation set
    # val_protomaml_sampler = TaskBatchSampler(
    #     val_set.tasks,
    #     val_set.labels,
    #     include_query=True,
    #     K_shot=K_SHOT,
    #     batch_size=1,  # We do not update the parameters, hence the batch size is irrelevant here
    #     shuffle=False
    # )
    # print('Creating val dataloader...')
    # val_protomaml_loader = data.DataLoader(
    #     val_set,
    #     batch_sampler=val_protomaml_sampler,
    #     collate_fn=val_protomaml_sampler.get_collate_fn(),
    #     num_workers=num_workers,
    # )


