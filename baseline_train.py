from gc import callbacks
from torch.utils.data import DataLoader
from baseline_module import BaselineModel
from baseline_dataset import MultiTaskDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import torch 

# import required to enable unpickling Argument dataset
from dataset_loader import ArgumentDatasetSplit

import pytorch_lightning as pl

adapter_tasks = ['mnli', 'qqp', 'sst', 'wgrande', 'boolq']
train_tasks = ['imdb', 'mrpc', 'argument', 'scitail']
k = 4
# using k=8, batch_size=1 and 4 tasks, the model sees 8*2*4*1=64 examples
# per update step (assuming 2 classes per task), which should be sufficient.
batch_size=1

print('adapters', adapter_tasks)
print('train_tasks', train_tasks)
print('k', k)
print('batch_size', batch_size)

dataloader = DataLoader(MultiTaskDataset(train_tasks, k=k), batch_size=batch_size)

model = BaselineModel(adapter_tasks, train_tasks, k=k)

# save model checkpoint at lowest training loss
checkpoint_callback = ModelCheckpoint(monitor="train_loss")

trainer = pl.Trainer(
    gpus=1 if torch.cuda.is_available() else 0, 
    max_epochs=1, 
    callbacks=[ModelCheckpoint(monitor="train_loss")],
    progress_bar_refresh_rate=100
)

trainer.fit(model=model, train_dataloaders=dataloader)