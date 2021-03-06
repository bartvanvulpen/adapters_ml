import os
import argparse
from gc import callbacks
from torch.utils.data import DataLoader
from baseline_module import BaselineModel
from baseline_dataset import MultiTaskDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from experiments import experiments

# import required to enable unpickling Argument dataset
from dataset_loader import ArgumentDatasetSplit

import pytorch_lightning as pl

if __name__ == "__main__":
    k = 4
    # using k=2, batch_size=1 and 4 tasks, the model sees 2*2*4*1=16 examples
    # per update step (assuming 2 classes per task)
    batch_size=1

    parser = argparse.ArgumentParser()

    parser.add_argument('--tasks', nargs="+", default=["1", "3", "4", "5", "6", "7", "8", "9"],
                        help='Experiments numbers to train', required=False)
    parser.add_argument('--k', default=4, type=int,
                        help='Number of datapoints per label during testing', required=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size', required=False)


    args = parser.parse_args()

    for experiment_str in args.tasks:
        experiment = int(experiment_str)

        adapter_tasks = experiment['adapter_tasks']
        train_tasks = experiment['train_tasks']
        exp_num = experiment['exp_number']

        print('------------------------------')
        print('START EXPERIMENT', exp_num)
        print('adapters', adapter_tasks)
        print('train_tasks', train_tasks)
        print('k =', args.k)
        print('batch_size =', args.batch_size)

        dataloader = DataLoader(MultiTaskDataset(train_tasks, k=args.k), batch_size=args.batch_size)

        model = BaselineModel(adapter_tasks, train_tasks, k=args.k, exp_num=exp_num)

        # save the model with the lowest training loss
        checkpoint = ModelCheckpoint(
            monitor='train_loss',
            every_n_train_steps=5,
            filename='Experiment=' + str(exp_num)
        )

        trainer = pl.Trainer(
            gpus=1 if torch.cuda.is_available() else 0,
            max_epochs=1,
            callbacks=[checkpoint],
            progress_bar_refresh_rate=100,
            log_every_n_steps=1
        )

        trainer.fit(model=model, train_dataloaders=dataloader)
