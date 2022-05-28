from baseline_dataset import collate_fn
from torch.utils.data import DataLoader
from dataset_loader import load_dataset_from_file
from baseline_module import BaselineModel
import pytorch_lightning as pl
import torch

torch.manual_seed(0)

def dict_to_device(dic, device):
    """
    sends a dict whose values are tensors, to device
    """
    for key in dic:
        dic[key] = dic[key].to(device)

# we perform the hyperparameter search using experiment 1

# we look for the optimal number of times to perform an update step on 
# the same few-shot train data
# this list contains all n_updates that we will test
updates_options = [2,4,6,10,15,20]

test_task = 'sick'
dataset, id2label = load_dataset_from_file(test_task)
n_classes = len(id2label)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

for n_updates in updates_options:

    print('------------------')
    print('Test value n_updates=', n_updates)

    model = BaselineModel.load_from_checkpoint('models/Experiment=1-step=454.ckpt')
    model.to(device)

    k = model.hparams.k

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
    model.train()
    model.prepare_for_test(test_task)

    optimizer = model.configure_optimizers()

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
    print('Accuracy', accuracy)
