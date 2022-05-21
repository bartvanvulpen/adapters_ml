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
from data_construction import get_test_loaders
from sampler import split_batch
from tqdm import tqdm
from ProtoMAML import ProtoMAML
from sampler import FewShotBatchSampler

## Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "checkpoints_meta_learning/"

def test_protomaml(model, task, k_shot=4, max_it=20, full_dl_batch_size=8):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pl.seed_everything(42)
    model = model.to(device)

    # get test dataloaders and sampler
    full_test_loader, sample_test_loader, sampler = get_test_loaders(task, K_SHOT=k_shot, full_dl_batch_size=full_dl_batch_size, num_workers=0)

    # Select the k-shot batch and finetune
    accuracies = []
    i = 0
    for x, support_indices in tqdm(zip(sample_test_loader, sampler), "Performing few-shot finetuning"):
        i += 1
        support_inputs = {'input_ids' : x[1][0].to(device), 'token_type_ids' : x[1][1].to(device),
                          'attention_mask' : x[1][2].to(device)}
        support_targets = x[2].to(device)

        # Finetune new model on support set
        local_model, output_weight, output_bias, classes = model.adapt_few_shot(
            support_inputs, support_targets
        )

        # get accuracy of finetuned model on full dataset
        with torch.no_grad():
            local_model.eval()
            batch_acc = torch.zeros((0,), dtype=torch.float32, device=device)

            for q_data in full_test_loader:

                query_inputs = {'input_ids' : q_data[1][0].to(device), 'token_type_ids' : q_data[1][1].to(device),
                                'attention_mask' : q_data[1][2].to(device)}

                query_targets = q_data[2].to(device)
                query_labels = (
                    (classes[None, :] == query_targets[:, None]).long().argmax(dim=-1)
                )
                _, _, acc = model.run_model(
                    local_model, output_weight, output_bias, query_inputs, query_labels
                )
                batch_acc = torch.cat([batch_acc, acc.detach()], dim=0)

            # exclude support set elements
            for s_idx in support_indices:
                batch_acc[s_idx] = 0
            batch_acc = batch_acc.sum().item() / (
                batch_acc.shape[0] - len(support_indices)
            )
            accuracies.append(batch_acc)

        # return mean accuracy over the runs after max iterations
        if i == max_it:
            return mean(accuracies), stdev(accuracies)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_task', type=str,
                        help='task to use for meta testing', required=True)

    parser.add_argument('--ckpt_path', type=str, default='metatrain_outputs/cb_mnli_sst-boolq_mrpc/lightning_logs/version_3/checkpoints/epoch=0-step=1.ckpt',
                        help='specifiy checkpoint path of model to use')

    parser.add_argument('--inner_steps', type=int, default=200,
                        help='number of inner steps')

    parser.add_argument('--max_it', type=int, default=20,
                        help='max number of few shot samplings in meta testing')

    parser.add_argument('--bs', type=int, default=8,
                        help='batch size for when iterating through the whole validation set to obtain accuracy')

    parser.add_argument('--results_file_path', type=str, default='result_file.json',
                        help='path + filename to store results file')

    parser.add_argument('--k_values', nargs="+", default=[2],
                        help='which k-values to test on')

    args = parser.parse_args()

    protomaml_result_file = args.results_file_path

    # Load specified checkpoint after training
    protomaml_model = ProtoMAML.load_from_checkpoint(args.ckpt_path)

    # change number of innersteps
    protomaml_model.hparams.num_inner_steps = args.inner_steps

    # Perform experiments
    protomaml_accuracies = dict()
    for k in args.k_values:
        protomaml_accuracies[k] = test_protomaml(protomaml_model, args.test_task, k_shot=k, max_it=args.max_it, full_dl_batch_size=args.bs)
    # Export results
    with open(protomaml_result_file, "w") as f:
        json.dump(protomaml_accuracies, f, indent=4)

    for k in protomaml_accuracies:
        print(
            f"Accuracy for k={k}: {100.0*protomaml_accuracies[k][0]:4.2f}% (+-{100.0*protomaml_accuracies[k][1]:4.2f}%)"
        )
