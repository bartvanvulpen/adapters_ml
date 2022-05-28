import torch
from torch.utils.data import DataLoader
from baseline_module import BaselineModel
from baseline_dataset import MultiTaskDataset
from dataset_loader import ArgumentDatasetSplit

import pytorch_lightning as pl

adapter_tasks = ['mnli', 'sst']
train_tasks = ['argument']

dataloader = DataLoader(MultiTaskDataset(train_tasks, k=3), batch_size=2)

model = BaselineModel(adapter_tasks, train_tasks)

trainer = pl.Trainer()
trainer.fit(model=model, train_dataloaders=dataloader)