## Standard libraries
import os
from copy import deepcopy
from statistics import mean, stdev
import json
import argparse
import os
import sys
## PyTorch
from adapter_fusion import load_bert_model
from transformers.adapters.composition import Fuse
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from transformers.adapters import BertAdapterModel
## PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset_loader import ArgumentDatasetSplit

## Dataset and sampler
from data_construction import get_train_val_loaders
from sampler import split_batch
from tqdm import tqdm
from ProtoMAML import ProtoMAML
from sampler import FewShotBatchSampler

## Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "checkpoints_meta_learning/"


def test_protomaml(model, dataset, k_shot=4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pl.seed_everything(42)
    model = model.to(device)
    num_classes = dataset.targets.unique().shape[0]
    exmps_per_class = dataset.targets.shape[0] // num_classes

    # Data loader for full test set as query set
    full_dataloader = data.DataLoader(
        dataset, batch_size=16, num_workers=4, shuffle=False, drop_last=False
    )
    # Data loader for sampling support sets
    sampler = FewShotBatchSampler(
        dataset.targets,
        include_query=False,
        N_way=num_classes,
        K_shot=k_shot,
        shuffle=False,
        shuffle_once=False,
    )
    sample_dataloader = data.DataLoader(dataset, batch_sampler=sampler, num_workers=2)

    # We iterate through the full dataset in two manners. First, to select the k-shot batch.
    # Second, the evaluate the model on all other examples
    accuracies = []
    for (support_imgs, support_targets), support_indices in tqdm(
        zip(sample_dataloader, sampler), "Performing few-shot finetuning"
    ):
        support_imgs = support_imgs.to(device)
        support_targets = support_targets.to(device)

        # Finetune new model on support set
        local_model, output_weight, output_bias, classes = model.adapt_few_shot(
            support_imgs, support_targets
        )
        with torch.no_grad():  # No gradients for query set needed
            local_model.eval()
            batch_acc = torch.zeros((0,), dtype=torch.float32, device=device)

            # Evaluate all examples in test dataset
            for query_imgs, query_targets in full_dataloader:
                query_imgs = query_imgs.to(device)
                query_targets = query_targets.to(device)
                query_labels = (
                    (classes[None, :] == query_targets[:, None]).long().argmax(dim=-1)
                )
                _, _, acc = model.run_model(
                    local_model, output_weight, output_bias, query_imgs, query_labels
                )
                batch_acc = torch.cat([batch_acc, acc.detach()], dim=0)

            # Exclude support set elements
            for s_idx in support_indices:
                batch_acc[s_idx] = 0
            batch_acc = batch_acc.sum().item() / (
                batch_acc.shape[0] - len(support_indices)
            )
            accuracies.append(batch_acc)

    return mean(accuracies), stdev(accuracies)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_tasks', nargs="+", default=['sst', 'boolq'],
                        help='task to use for meta_training for')


    # TODO: protomaml model

    # Load best checkpoint after training

    path_to_best_model = "xxx"
    model_class = ProtoMAML
    protomaml_model = model_class.load_from_checkpoint(path_to_best_model)


    # TODO: test set dataloader
    test_set = []

    protomaml_result_file = os.path.join(CHECKPOINT_PATH, "protomaml_fewshot.json")


    if os.path.isfile(protomaml_result_file):
        # Load pre-computed results
        with open(protomaml_result_file, "r") as f:
            protomaml_accuracies = json.load(f)
        protomaml_accuracies = {int(k): v for k, v in protomaml_accuracies.items()}
    else:
        # Perform experiments
        protomaml_accuracies = dict()
        for k in [2, 4, 8, 16, 32]:
            protomaml_accuracies[k] = test_protomaml(protomaml_model, test_set, k_shot=k)
        # Export results
        with open(protomaml_result_file, "w") as f:
            json.dump(protomaml_accuracies, f, indent=4)

    for k in protomaml_accuracies:
        print(
            f"Accuracy for k={k}: {100.0*protomaml_accuracies[k][0]:4.2f}% (+-{100.0*protomaml_accuracies[k][1]:4.2f}%)"
        )
