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
            self.classes[t] = torch.unique(self.dataset_targets[torch.where(self.dataset_tasks == t)[0]]).tolist()
            self.num_classes[t] = len(self.classes[t])
            self.indices_per_class[t] = {}
            self.batches_per_class[t] = {}  # Number of K-shot batches that each class can provide
            for c in self.classes[t]:
                self.indices_per_class[t][c] = torch.where(
                    (self.dataset_tasks == t) *
                    (self.dataset_targets == c)
                )[0]


                self.batches_per_class[t][c] = self.indices_per_class[t][c].shape[0] // self.K_shot
            self.batches_per_task[t] = sum(self.batches_per_class[t].values())

        self.unique_task_classes = [(t,c) for t in self.tasks for c in self.classes[t]]


        # Create a list of task-class tuples from which we select the N classes per batch
        self.iterations_per_task = [sum(self.batches_per_class[t].values()) // self.N_way for t in self.tasks]
        self.task_list = [t for i, t in enumerate(self.tasks) for _ in range(self.iterations_per_task[i])]
        self.iterations = sum(self.iterations_per_task)

        self.class_list = {
            t: [c for c in self.classes[t] for _ in range(self.batches_per_class[t][c])]
            for t in self.task_list
        }
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

        for it in range(self.iterations):

            t = random.choice(self.tasks)

            # For each task-class tuple, select the next K examples and add them to the batch
            index_batch = []
            for c in self.classes[t]:
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
            batch_list.append(batch)
            if (batch_idx+1) % self.task_batch_size == 0:
                yield batch_list
                batch_list = []

    def __len__(self):
        return len(self.batch_sampler)//self.task_batch_size

   
    def get_collate_fn(self):
        # Returns a collate function that converts a list of items into format for transformer model
        def collate_fn(item_list):
            print(len(item_list))
            print(item_list[0])
            collated = [(x, label) for _, x, label in item_list]
            return collated

        # TODO: insert collate_fn for multiple choice from Github



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




