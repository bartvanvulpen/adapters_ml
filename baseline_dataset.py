from numpy import iterable
import torch
from torch.utils.data import DataLoader
from dataset_loader import load_dataset_from_file
from dataset_loader import ArgumentDatasetSplit

class InfiniteIterator():
    def __init__(self, iterable):
        self.iterable = iterable
        self.iter = iter(iterable)

    def __next__(self):
        next_item = next(self.iter, None)

        if next_item is None:
            # this happens when the iterator is exhausted.
            # we restart iteration and fetch the next (i.e. first) item
            self.iter = iter(self.iterable)
            next_item = next(self.iter)

        return next_item


def collate_fn(batch_list):
    d = {}
    for key in ['input_ids', 'token_type_ids', 'attention_mask', 'labels']:
        d[key] = torch.vstack([torch.tensor(x[key]) for x in batch_list])

    return d


class MultiTaskDataset():
    def __init__(self, task_names_list, k):
        """
        task_names_list - list containing the names of the tasks. Should be the name as
        used in dataset_loader.py
        k - the k value used in the meta-training setup for which this model will 
        be a baseline. This influences the number of training examples retrieved
        """
        super().__init__()

        # contains iterators over all tasks
        self.datasets = []

        for task_name in task_names_list:
            dataset, id2label = load_dataset_from_file(task_name)
            n_classes = len(id2label)
            dataset = InfiniteIterator(DataLoader(
                dataset['train'], 
                batch_size=k * n_classes, 
                shuffle=True,
                collate_fn=collate_fn
            ))

            self.datasets.append({
                'name': task_name,
                'dataset': dataset
            })


    def __len__(self):
        return 200 * 16 // len(self.datasets)


    def __getitem__(self, idx):
        item = {}
        for dataset in self.datasets:
            item[dataset['name']] = next(dataset['dataset'])

        return item

