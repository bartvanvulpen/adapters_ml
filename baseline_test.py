# import required to enable unpickling Argument dataset
from dataset_loader import ArgumentDatasetSplit

from unittest import result
from baseline_dataset import collate_fn
from torch.utils.data import DataLoader
from dataset_loader import load_dataset_from_file
from baseline_module import BaselineModel
from data_construction import get_validation_split_name
import pytorch_lightning as pl
import torch
from experiments import experiments
import pickle
import argparse
import numpy as np
import json

def dict_to_device(dic, device):
    """
    sends a dict whose values are tensors, to device
    """
    for key in dic:
        dic[key] = dic[key].to(device)


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    # using k=2, batch_size=1 and 4 tasks, the model sees 2*2*4*1=16 examples
    # per update step (assuming 2 classes per task)
    batch_size=1

    parser = argparse.ArgumentParser()

    parser.add_argument('--tasks', nargs="+", default=["1", "3", "4", "5", "6", "7", "8", "9"],
                        help='Experiments numbers to train', required=False)
    parser.add_argument('--n_updates', default=10, type=int,
                        help='Number of update steps', required=False)
    parser.add_argument('--outfile', default='baseline_test_results.json', type=str,
                        help='File to save the results to', required=False)

    args = parser.parse_args()

    # this list will contain all results
    results = []

    for experiment_str in args.tasks:
        experiment = None
        for exp in experiments:
            if exp['exp_number'] == int(experiment_str):
                experiment = exp
                break
        if not experiment:
            print('Exp number not found!')
            break

        exp_num = experiment['exp_number']
        test_tasks = experiment['test_tasks']

        for test_task in test_tasks:
            for k in [2,4,8]:
                print('------------------')
                print('Start new evaluation with values:')
                print('Experiment number =', exp_num)
                print('Test task =', test_task)
                print('k =', k)
                all_accuracies = []
                for i in range(20):
                    torch.manual_seed(i)

                    dataset, id2label = load_dataset_from_file(test_task)
                    n_classes = len(id2label)

                    model = BaselineModel.load_from_checkpoint('models/Experiment=' + str(exp_num)+'.ckpt')

                    key = get_validation_split_name(test_task)
                    dl = DataLoader(dataset[key], #Take the validation split
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
                    for _ in range(args.n_updates):
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
                    all_accuracies.append(accuracy)


                    print(f'iter {i} | accuracy: {accuracy}')

                print(f'avg: {np.average(all_accuracies)}')
                print(f'avg: {np.std(all_accuracies)}')
                results.append({
                    'exp_number': exp_num,
                    'test_task': test_task,
                    'k': k,
                    'accuracy': np.average(all_accuracies),
                    'std': np.std(all_accuracies)
                })

                # save the results every iteration so we can recover
                # results if an error occurs
                with open(args.outfile, 'w') as f:
                    json.dump(results, f)
