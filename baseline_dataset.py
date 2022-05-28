import torch
from torch.utils.data import DataLoader
from dataset_loader import load_dataset_from_file

# import required to enable unpickling Argument dataset
from dataset_loader import ArgumentDatasetSplit


class InfiniteIterator():
    """
    This iterator cycles through an iterable ad infinitum.
    When the iterable is exhausted, this iterator just restarts
    iteration at the first item and continues. 
    """
    def __init__(self, iterable):
        self.iterable = iterable
        self.iter = iter(iterable)

    def __next__(self):
        # None is a default value, returned when the iterable is exhausted
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
    def __init__(self, train_tasks, k):
        """
        train_tasks - list containing the names of the tasks.
        all task names: ['mnli','qqp','sst','wgrande','imdb','scitail','argument','boolq','mrpc','sick','rte','cb']
        k - the k value used in the meta-training setup for which this model will 
        be a baseline. This influences the number of training examples retrieved.
        k should be 2, 4, or 8

        After 1 epoch, this Dataset has returned exactly as many data points as
        the corresponding meta-train setup for which this model is a baseline.
        However, in its current implementation, this Dataset will return 
        new, different items when a second epoch is started. Hence, for now, this
        Dataset should not be used for more than a single epoch. 
        """
        super().__init__()

        # contains iterators over all tasks
        self.datasets = []

        self.k = k

        for task_name in train_tasks:
            dataset, id2label = load_dataset_from_file(task_name)
            n_classes = len(id2label)

            # Every task returns k * n_classes items per call of __getitem__.
            # This is in agreement with the meta-train setup, where the model
            # sees more examples of tasks with more classes. 
            # However, to avoid memory problems, we always return 2*n_classes
            # items per batch. We correct for k in the __len__. Larger
            # k leads to a greater dataset, and hence more training examples
            # in total. Remember this Dataset is supposed to be used for 
            # just a single epoch.
            dataset = InfiniteIterator(DataLoader(
                dataset['train'], 
                batch_size=2 * n_classes, 
                shuffle=True,
                collate_fn=collate_fn
            ))

            self.datasets.append({
                'name': task_name,
                'dataset': dataset
            })


    def __len__(self):
        return 200 * 16 * (self.k / 2) // len(self.datasets)


    def __getitem__(self, idx):
        # __getitem__ is used differently here to how it's conventionally done.
        # this method is supposed to return a specific index of a list-type.
        # however, we just return the next item of each dataset, and ignore
        # the idx argument. Shuffling still happens, because the datasets
        # contained in this MultiTaskDataset are all individually shuffled.
        item = {}
        for dataset in self.datasets:
            item[dataset['name']] = next(dataset['dataset'])

        return item

