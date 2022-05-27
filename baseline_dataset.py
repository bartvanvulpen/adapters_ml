from numpy import iterable
import torch
from torch.utils.data import DataLoader, IterableDataset
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
            dataset, id2label = load_dataset_from_file('argument')
            n_classes = len(id2label)
            dataset = InfiniteIterator(DataLoader(
                dataset['train'], 
                batch_size=n_classes, 
                shuffle=True
            ))

            self.datasets.append({
                'name': task_name,
                dataset: dataset
            })


    def __len__():
        return 200 * 16 * k / num_tasks

    def __getitem__(self):
        return {
            'argument': next(self.ds1),
            'boolq': next(self.ds2)
        }

# dl = DataLoader(MultiTaskDataset())

# i=0
# for x in dl:
#     print(x)
#     i+=1
#     if i == 5:
#         break


# def multi_task_collator(sample_list):