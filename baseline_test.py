from baseline_dataset import collate_fn
from torch.utils.data import DataLoader
from dataset_loader import load_dataset_from_file
from baseline_module import BaselineModel
import pytorch_lightning as pl
import torch

torch.manual_seed(0)

adapter_tasks = ['mnli', 'qqp', 'sst', 'wgrande', 'boolq']
train_tasks = ['imdb', 'mrpc', 'argument', 'scitail']

model = BaselineModel.load_from_checkpoint(
    'models/epoch=0-step=399-v1.ckpt',
    adapter_tasks=adapter_tasks, train_tasks=train_tasks, lr=1e-5)

test_task = 'sick'
dataset, id2label = load_dataset_from_file(test_task)
n_classes = len(id2label)
k = 2

dl = DataLoader(dataset['test'], 
        batch_size=k * n_classes, 
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
)

data_iterator = iter(dl)


# get the first batch to train the model
train_batch = next(data_iterator)
model.train()
model.prepare_for_test(test_task)

optimizer = model.configure_optimizers()

for i in range(6):
    optimizer.zero_grad()
    loss = model.train_step_single_task(train_batch)
    loss.backward()
    optimizer.step()


model.eval()
accuracies = []

i=0
# loop over the remaining batches to compute accuracy
for batch in data_iterator:
    accuracies.append(model.compute_accuracy(batch))

    i+=1
    if i == 10:
        break

accuracy = sum(accuracies) / len(accuracies)
print('Accuracy', accuracy)
