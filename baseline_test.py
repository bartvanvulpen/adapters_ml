from unittest import result
from baseline_dataset import collate_fn
from torch.utils.data import DataLoader
from dataset_loader import load_dataset_from_file
from baseline_module import BaselineModel
import pytorch_lightning as pl
import torch
from experiments import experiments
import pickle

torch.manual_seed(0)

def dict_to_device(dic, device):
    """
    sends a dict whose values are tensors, to device
    """
    for key in dic:
        dic[key] = dic[key].to(device)


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# we perform 10 update steps on the few-shot train data. This number was found to work
# well for all k values (tested in a hyperparam search)
n_updates = 10

# this list will contain all results
results = []

for experiment in experiments:

    exp_num = experiment['exp_number']
    test_tasks = experiment['test_tasks']

    for test_task in test_tasks:
        for k in [2,4,8]:

            print('------------------')
            print('Start new evaluation with values:')
            print('Experiment number =', exp_num)
            print('Test task =', test_task)
            print('k =', k)
            

            dataset, id2label = load_dataset_from_file(test_task)
            n_classes = len(id2label)

            model = BaselineModel.load_from_checkpoint('models/Experiment=' + str(exp_num)+'.ckpt')

            dl = DataLoader(dataset['test'], 
                    batch_size=k * n_classes, 
                    shuffle=True,
                    collate_fn=collate_fn,
                    drop_last=True
            )

            data_iterator = iter(dl)

            # get the first batch to train the model
            train_batch = next(data_iterator)
            dict_to_device(train_batch, device)
            
            model.prepare_for_test(test_task)
            model.to(device)

            optimizer = model.configure_optimizers()

            model.train()
            for _ in range(n_updates):
                optimizer.zero_grad()
                loss = model.train_step_single_task(train_batch)
                loss.backward()
                optimizer.step()


            model.eval()
            accuracies = []

            # loop over the remaining batches to compute accuracy
            for batch in data_iterator:
                dict_to_device(batch, device)
                accuracies.append(model.compute_accuracy(batch))


            accuracy = sum(accuracies) / len(accuracies)
            print('Accuracy =', accuracy)

            results.append({
                'exp_number': exp_num,
                'test_task': test_task,
                'k': k,
                'accuracy': accuracy
            })

            # save the results every iteration so we can recover
            # results if an error occurs
            with open('test_results.pickle', 'wb') as f:
                pickle.dump(results, f)

