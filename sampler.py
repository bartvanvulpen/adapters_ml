## Standard libraries
import numpy as np
import random
from collections import defaultdict


## PyTorch
import torch
import torch.utils.data as data


from dataloader import load_mnli, load_qqp, load_sst2, load_boolq, load_cb

## Define the datasets that we are using for metalearning
DATALOADERS = {
    "mnli": load_mnli(),
    "qqp": load_qqp(),
    "sst": load_sst2(),
#    "wgrande": 
    "boolq": load_boolq(),
#    "imdb": 
#    "hswag":
#    "mrpc":
#    "argument": 
#    "scitail": 
#    "sociqa": 
#    "cosqa": 
#    "csqa": 
#    "sick": 
#    "rte": 
    "cb": load_cb()
}


def combine_train_valid(name):
    """
    Combine the train and validation sets into one big dataset
    """
    
    (ds, id2label), key = DATALOADERS[name]

    # combine input data from train and validation set
    all_inputs = torch.cat((ds["train"]["input_ids"], ds[key]["input_ids"]), dim=0)
    all_token_types = torch.cat((ds["train"]["token_type_ids"], ds[key]["token_type_ids"]), dim=0)
    all_masks = torch.cat((ds["train"]["attention_mask"], ds[key]["attention_mask"]), dim=0)
    all_labels = torch.cat((ds["train"]["labels"], ds[key]["labels"]), dim=0)
    
    return {
        "input_ids": all_inputs, 
        "token_type_ids": all_token_types, 
        "attention_mask": all_masks,
        "labels": all_labels
    }

DATASETS = {ds: combine_train_valid(ds) for ds in DATALOADERS.keys()}
TASK_IDS = {name: id for id, name in enumerate(DATASETS.keys())}


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
        return self.inputs.shape[0]


def dataset_from_tasks(dataset, tasks, **kwargs):
    """
    Create a new dataset based on filtering out specific tasks in the combined dataset
    """
    
    task_mask = (dataset["tasks"][:,None] == tasks[None,:]).any(dim=-1)
    dataset = NLPDataset(
        tasks = dataset["tasks"][task_mask], 
        input_ids = dataset["input_ids"][task_mask], 
        token_type_ids = dataset["token_type_ids"][task_mask], 
        attention_mask = dataset["attention_mask"][task_mask], 
        labels = dataset["labels"][task_mask], 
        **kwargs
    )
    return dataset


class FewShotBatchSampler(object):
    """
    Sampler based on the sampler in the DL tutorial about meta-learning.
    This sampler is adjusted to enable sampling based a specific tasks and classes.
    """

    def __init__(self, dataset_tasks, dataset_targets, N_way, K_shot, include_query=False, shuffle=True, shuffle_once=False):
        """
        Inputs:
            dataset_tasks - PyTorch tensor of the id's of the tasks in the dataset.
            dataset_classes - PyTorch tensor of the classes in the dataset
            N_way - Number of classes to sample per batch.
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
        """
        super().__init__()
        self.dataset_tasks = dataset_tasks
        self.dataset_targets = dataset_targets
        self.dataset_task_targets = torch.cat((dataset_tasks.unsqueeze(dim=1), dataset_targets.unsqueeze(dim=1)), dim=1)
        self.N_way = N_way
        self.K_shot = K_shot
        self.shuffle = shuffle
        self.include_query = include_query
        if self.include_query:
            self.K_shot *= 2
        self.batch_size = self.N_way * self.K_shot  # Number of overall samples per batch

        # Organize examples by task and class
        self.tasks = torch.unique(self.dataset_tasks).tolist()
        self.classes = {}
        self.num_classes = {}
        self.indices_per_task = {}
        self.batches_per_task = {}
        self.indices_per_class = {}
        self.batches_per_class = {}
        for t in self.tasks:
            self.indices_per_task[t] = torch.where(self.dataset_tasks == t)[0].tolist()
#            print("Indices for task {}: {}".format(t, self.indices_per_task[t]))
#            print("Classes for task {}: {}".format(t, torch.unique(self.dataset_targets[self.indices_per_task[t]]).tolist()))
            self.classes[t] = torch.unique(self.dataset_targets[torch.where(self.dataset_tasks == t)[0]]).tolist()
            self.num_classes[t] = len(self.classes[t])
            self.indices_per_class[t] = {}
            self.batches_per_class[t] = {}  # Number of K-shot batches that each class can provide
            for c in self.classes[t]:
                self.indices_per_class[t][c] = torch.where(
                    (self.dataset_tasks == t) *
                    (self.dataset_targets == c)
                )[0]
                print("Indices per class ({}, {}): {}".format(t, c, self.indices_per_class[t][c]))
                self.batches_per_class[t][c] = self.indices_per_class[t][c].shape[0] // self.K_shot
#                print("Batches per class ({}, {}): {}".format(t, c, self.batches_per_class[t][c]))
            self.batches_per_task[t] = sum(self.batches_per_class[t].values())
#            print("Batches for task {}: {}".format(t, self.batches_per_task[t]))
        self.unique_task_classes = [(t,c) for t in self.tasks for c in self.classes[t]]

        # Create a list of task-class tuples from which we select the N classes per batch
        self.iterations_per_task = [sum(self.batches_per_class[t].values()) // self.N_way for t in self.tasks]
        self.task_list = [t for t in self.tasks for _ in range(self.iterations_per_task[t])]
        print("Task_list  (init): ", self.task_list)
        self.iterations = sum(self.iterations_per_task)
#        print("Iterations: ", self.iterations_per_task)
        self.class_list = {
            t: [c for c in self.classes[t] for _ in range(self.batches_per_class[t][c])]
            for t in self.task_list
        }
        print("Class_list (init): ", self.class_list)
        if shuffle_once or self.shuffle:
            self.shuffle_data()
        else:
            # For testing, we iterate over tasks and classes instead of shuffling them
            for t in self.tasks:
                sort_idxs = [
                    i + p * self.num_classes[t]
                    for i, c in enumerate(self.classes[t]) 
                    for p in range(self.batches_per_class[t][c])
                ]
                self.class_list[t] = np.array(self.class_list[t])[np.argsort(sort_idxs)].tolist()
        print("Class_list (final): ", self.class_list)
        print("Task_list  (final): ", self.task_list)
            
    def shuffle_data(self):
        # Shuffle the examples per task and class.       
        for t,c in self.unique_task_classes:
            perm = torch.randperm(self.indices_per_class[t][c].shape[0])
            self.indices_per_class[t][c] = self.indices_per_class[t][c][perm]

        # Shuffle the order of the tasks
        random.shuffle(self.task_list)
        
        # Lastly, shuffle the class list per task.
        # Note that this way of shuffling does not prevent to choose the same class twice in a batch. 
        # Especially with NLP-tasks with small number of classes, this happens quite often  
        for t in self.tasks:
            random.shuffle(self.class_list[t])

    def __iter__(self):
        # Shuffle data
        if self.shuffle:
            self.shuffle_data()

        # Sample few-shot batches
        start_index = defaultdict(int)
        task_iter = [0] * len(self.tasks)
        for it in range(self.iterations):
            
            # Select N classes for task t for the batch
            t = self.task_list[it]
            idx = task_iter[t] * self.N_way
            task_iter[t] += 1
            class_batch = self.class_list[t][idx:idx+self.N_way]

            # For each task-class tuple, select the next K examples and add them to the batch
            index_batch = []
            for c in class_batch:  
                index_batch.extend(self.indices_per_class[t][c][start_index[t,c]:start_index[t,c]+self.K_shot])
                start_index[t,c] += self.K_shot
                
            # If we return support+query set, sort them so that they are easy to split
            if self.include_query:
                index_batch = index_batch[::2] + index_batch[1::2]
            yield [i.item() for i in index_batch]

    def __len__(self):
        return self.iterations

class TaskBatchSampler(object):

    def __init__(self, dataset_tasks, dataset_targets, batch_size, N_way, K_shot, include_query=False, shuffle=True):
        """
        Inputs:
            dataset_tasks - PyTorch tensor of the id's of the tasks in the dataset.
            dataset_classes - PyTorch tensor of the classes in the dataset
            batch_size - Number of tasks to aggregate in a batch
            N_way - Number of classes to sample per batch.
            K_shot - Number of examples to sample per class in the batch.
            include_query - If True, returns batch of size N_way*K_shot*2, which
                            can be split into support and query set. Simplifies
                            the implementation of sampling the same classes but
                            distinct examples for support and query set.
            shuffle - If True, examples and classes are newly shuffled in each
                      iteration (for training)
        """
        super().__init__()
        self.batch_sampler = FewShotBatchSampler(dataset_tasks, dataset_targets, N_way, K_shot, include_query, shuffle)
        self.task_batch_size = batch_size
        self.local_batch_size = self.batch_sampler.batch_size

    def __iter__(self):
        # Aggregate multiple batches before returning the indices
        batch_list = []
        for batch_idx, batch in enumerate(self.batch_sampler):
            batch_list.extend(batch)
            if (batch_idx+1) % self.task_batch_size == 0:
                yield batch_list
                batch_list = []

    def __len__(self):
        return len(self.batch_sampler)//self.task_batch_size

    
    def get_collate_fn(self):
        # Returns a collate function that converts a list of items into format for transformer model
        
        def collate_fn(item_list):
            input_batch = {
                "input_ids": torch.stack([x[0] for task, x, label in item_list], dim=0),
                "token_type_ids": torch.stack([x[1] for task, x, label in item_list], dim=0),
                "attention_mask": torch.stack([x[2] for task, x, label in item_list], dim=0)
            } 
            label_batch = torch.stack([label for task, x, label in item_list], dim=0)
            return input_batch, label_batch
        
        return collate_fn


def split_batch(inputs, targets):
    """
    Split inputs and targets in two batches.
    Format needs to match with requirements of adaptorfusion model
    """
    
    support_input_ids, query_input_ids = inputs["input_ids"].chunk(2, dim=0)
    support_token_type_ids, query_token_type_ids = inputs["token_type_ids"].chunk(2, dim=0)
    support_attention_mask, query_attention_mask = inputs["attention_mask"].chunk(2, dim=0)
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

