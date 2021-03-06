## Standard libraries
import numpy as np
import random
from collections import defaultdict

## PyTorch
import torch



class FewShotBatchSampler(object):
    """
    Sampler based on the sampler in the DL tutorial about meta-learning.
    This sampler is adjusted to enable sampling based a specific tasks and classes.
    """

    def __init__(self, dataset_tasks, dataset_targets, K_shot, include_query=False, shuffle=True, shuffle_once=False):
        """
        Inputs:
            dataset_tasks - PyTorch tensor of the id's of the tasks in the dataset.
            dataset_classes - PyTorch tensor of the classes in the dataset
            K_shot - Number of examples to sample per class in the batch.
            include_query - If True, returns batch of size N_way*K_shot*2, which 
                            can be split into support and query set. Simplifies
                            the implementation of sampling the same classes but 
                            distinct examples for support and query set.
            shuffle - If True, examples and classes are newly shuffled in each
                      iteration (for training)
            shuffle_once - If True, examples and classes are shuffled once in 
                           the beginning, but kept constant across iterations 
                           (for validation)
            NOTE: the number of classes sampled per batch is always equal to the total number of classes for that task
        """
        super().__init__()
        self.dataset_tasks = dataset_tasks
        self.dataset_targets = dataset_targets
        self.dataset_task_targets = torch.cat((dataset_tasks.unsqueeze(dim=1), dataset_targets.unsqueeze(dim=1)), dim=1)
        self.K_shot = K_shot
        self.shuffle = shuffle
        self.include_query = include_query
        if self.include_query:
            self.K_shot *= 2

        # Organize examples by task and class
        self.tasks = torch.unique(self.dataset_tasks).tolist()
        self.classes = {}
        self.indices_per_class = {}
        for t in self.tasks:
            self.classes[t] = torch.unique(self.dataset_targets[torch.where(self.dataset_tasks == t)[0]]).tolist()
            self.indices_per_class[t] = {}
            for c in self.classes[t]:
                self.indices_per_class[t][c] = torch.where(
                    (self.dataset_tasks == t) *
                    (self.dataset_targets == c)
                )[0]

        if shuffle_once or self.shuffle:
            self.shuffle_data()

            
    def shuffle_data(self):
        # Shuffle the examples per task and class.       
        for t in self.tasks:
            for c in self.classes[t]:
                perm = torch.randperm(self.indices_per_class[t][c].shape[0])
                self.indices_per_class[t][c] = self.indices_per_class[t][c][perm]


    def __iter__(self):
        # Shuffle data
        if self.shuffle:
            self.shuffle_data()

        # Remember current sampler index, per task and class
        start_index = defaultdict(int)

        # Infinite sampler
        while True:

            t = random.choice(self.tasks)

            # For each class in this task, select the next K examples and add them to the batch
            index_batch = []
            for c in self.classes[t]:
                index_batch.extend(self.indices_per_class[t][c][start_index[t,c]:start_index[t,c]+self.K_shot])
                start_index[t,c] += self.K_shot

                # Reset index when you reach end if the list
                if (start_index[t,c] + self.K_shot) > len(self.indices_per_class[t][c]):
                    start_index[t,c] = 0

            # If we return support+query set, sort them so that they are easy to split
            if self.include_query:
                index_batch = index_batch[::2] + index_batch[1::2]

            yield [i.item() for i in index_batch]

    def __len__(self):
        # Infinite sampler
        return 1e100


class TaskBatchSampler(object):

    def __init__(self, dataset_tasks, dataset_targets, batch_size, K_shot, include_query=False, shuffle=True):
        """
        Inputs:
            dataset_tasks - PyTorch tensor of the id's of the tasks in the dataset.
            dataset_classes - PyTorch tensor of the classes in the dataset
            batch_size - Number of tasks to aggregate in a batch
            K_shot - Number of examples to sample per class in the batch.
            include_query - If True, returns batch of size N_way*K_shot*2, which
                            can be split into support and query set. Simplifies
                            the implementation of sampling the same classes but
                            distinct examples for support and query set.
            shuffle - If True, examples and classes are newly shuffled in each
                      iteration (for training)
            NOTE: the number of classes sampled per batch is always equal to the total number of classes for that task
        """
        super().__init__()
        self.batch_sampler = FewShotBatchSampler(dataset_tasks, dataset_targets, K_shot, include_query, shuffle)
        self.task_batch_size = batch_size
        # self.local_batch_size = self.batch_sampler.batch_size

    def __iter__(self):
        # Aggregate multiple batches before returning the indices
        batch_list = []
        for batch_idx, batch in enumerate(self.batch_sampler):
            batch_list.append(batch)
            if (batch_idx+1) % self.task_batch_size == 0:
                yield batch_list
                batch_list = []

    def __len__(self):
        return self.batch_sampler.__len__()//self.task_batch_size

   
    def get_collate_fn(self):
        # Returns a collate function that converts a list of items into format for transformer model
        def collate_fn(item_list):
            collated = [(x, label) for _, x, label in item_list]
            return collated

        return collate_fn


def split_batch(inputs, targets):
    """
    Split inputs and targets in two batches.
    Format needs to match with requirements of adaptorfusion model
    """

    support_input_ids, query_input_ids = inputs[0].chunk(2, dim=0)
    support_token_type_ids, query_token_type_ids = inputs[1].chunk(2, dim=0)
    support_attention_mask, query_attention_mask = inputs[2].chunk(2, dim=0)
    support_inputs = {
        "input_ids": support_input_ids,
        "token_type_ids": support_token_type_ids,
        "attention_mask": support_attention_mask
    } 
    query_inputs = {
        "input_ids": query_input_ids,
        "token_type_ids": query_token_type_ids,
        "attention_mask": query_attention_mask
    }

    support_targets, query_targets = targets.chunk(2, dim=0)
    return support_inputs, query_inputs, support_targets, query_targets




