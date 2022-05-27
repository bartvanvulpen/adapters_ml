import torch
from torch.utils.data import DataLoader
from baseline_module import BaselineModel
from baseline_dataset import MultiTaskDataset
from dataset_loader import ArgumentDatasetSplit

import pytorch_lightning as pl

dataloader = DataLoader(MultiTaskDataset(['imdb'], 3), batch_size=2)
model = BaselineModel()

trainer = pl.Trainer()
trainer.fit(model=model, train_dataloaders=dataloader)