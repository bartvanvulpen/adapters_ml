# Standard libraries
import os
from copy import deepcopy
from statistics import mean, stdev
import json
import argparse
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

## Dataset and sampler
from data_construction import get_train_loader
from sampler import split_batch

"""
Load BERT with adapter fusion
"""

def get_adapter_fusion_model(adapters_to_use, adapterfusion_path = None):

    if adapters_to_use == []:
        raise RuntimeError('You must specifiy the adapters to use!')


    all_adapters = {"mnli": "nli/multinli@ukp", "qqp": "sts/qqp@ukp", "sst": "sentiment/sst-2@ukp",
                    "wgrande": "comsense/winogrande@ukp", "boolq": "qa/boolq@ukp", "imdb": "sentiment/imdb@ukp",
                    "scitail": "nli/scitail@ukp", "argument" : "argument/ukpsent@ukp", "mrpc" : "sts/mrpc@ukp",
                    "sick" : "nli/sick@ukp", "rte" : "nli/rte@ukp", "cb" : "nli/cb@ukp", "siqa": "comsense/siqa@ukp",
                    "csqa": "comsense/csqa@ukp", "cqa": "comsense/cosmosqa@ukp", "hswag": "comsense/hellaswag@ukp"}


    model = BertAdapterModel.from_pretrained("bert-base-uncased")
    for a in adapters_to_use:
        model.load_adapter(all_adapters[a], load_as=a, with_head=False, config="pfeiffer")
    adapter_setup = Fuse(*adapters_to_use)
    model.add_adapter_fusion(adapter_setup)
    model.set_active_adapters(adapter_setup)
    model.train_adapter_fusion(adapter_setup)

    print(f'Loading adaptefusion from path: {adapterfusion_path}')
    if adapterfusion_path:
        model.load_adapter_fusion(
            adapterfusion_path,
            load_as=",".join(adapters_to_use),
            set_active=True,
        )

    return model

"""
PROTOMAML MODEL (incl function to calculate prototypes)
"""

class ProtoMAML(pl.LightningModule):
    def __init__(self, lr, lr_inner, lr_output, num_inner_steps, k_shot, task_batch_size, tasks, adapters_used, adapterfusion_path=None):
        """
        Inputs
            proto_dim - Dimensionality of prototype feature space
            lr - Learning rate of the outer loop Adam optimizer
            lr_inner - Learning rate of the inner loop SGD optimizer
            lr_output - Learning rate for the output layer in the inner loop
            num_inner_steps - Number of inner loop updates to perform
        """
        super().__init__()

        self.save_hyperparameters()
        self.model = get_adapter_fusion_model(adapters_used, adapterfusion_path)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[140, 180], gamma=0.1
        )
        return [optimizer], [scheduler]

    def calculate_prototypes(self, features, targets):
        # Given a stack of features vectors and labels, return class prototypes
        # features - shape [N, proto_dim], targets - shape [N]
        classes, _ = torch.unique(targets).float().sort()  # Determine which classes we have
        prototypes = []

        for c in classes:
            p = features[torch.where(targets == c)[0]].mean(
                dim=0
            )  # Average class feature vectors
            prototypes.append(p)
        prototypes = torch.stack(prototypes, dim=0)
        # Return the 'classes' tensor to know which prototype belongs to which class
        return prototypes, classes

    def run_model(self, local_model, output_weight, output_bias, inputs, labels):

        # Execute a model with given output layer weights and inputs
        feats = local_model(input_ids=inputs['input_ids'],
                                   attention_mask=inputs['attention_mask'],
                                   token_type_ids=inputs['token_type_ids']).pooler_output
        preds = F.linear(feats, output_weight, output_bias)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float()

        return loss, preds, acc

    def adapt_few_shot(self, support_inputs, support_targets):

        # Determine prototype initialization
        support_feats = self.model(input_ids=support_inputs['input_ids'],
                                   attention_mask=support_inputs['attention_mask'],
                                   token_type_ids=support_inputs['token_type_ids']).pooler_output

        prototypes, classes = self.calculate_prototypes(support_feats, support_targets)

        support_labels = (
                (classes[None, :] == support_targets[:, None]).long().argmax(dim=-1)
        )

        # Create inner-loop model and optimizer
        local_model = deepcopy(self.model)
        local_model.train()
        local_optim = optim.SGD(local_model.parameters(), lr=self.hparams.lr_inner)
        local_optim.zero_grad()

        # Create output layer weights with prototype-based initialization
        init_weight = 2 * prototypes
        init_bias = -torch.norm(prototypes, dim=1) ** 2
        output_weight = init_weight.detach().requires_grad_()
        output_bias = init_bias.detach().requires_grad_()

        # Optimize inner loop model on support set
        for i in range(self.hparams.num_inner_steps):

            # Determine loss on the support set
            loss, _, _ = self.run_model(
                local_model, output_weight, output_bias, support_inputs, support_labels
            )

            # Calculate gradients and perform inner loop update
            loss.backward()
            local_optim.step()

            # Update output layer via SGD
            output_weight.data -= self.hparams.lr_output * output_weight.grad
            output_bias.data -= self.hparams.lr_output * output_bias.grad

            # Reset gradients
            local_optim.zero_grad()
            output_weight.grad.fill_(0)
            output_bias.grad.fill_(0)

        # Re-attach computation graph of prototypes
        output_weight = (output_weight - init_weight).detach() + init_weight
        output_bias = (output_bias - init_bias).detach() + init_bias

        return local_model, output_weight, output_bias, classes

    def outer_loop(self, batch, mode="train"):

        accuracies = []
        losses = []
        self.model.zero_grad()

        # Determine gradients for batch of tasks
        for i, task_batch in enumerate(batch):
            inputs, targets = task_batch

            support_inputs, query_inputs, support_targets, query_targets = split_batch(
                inputs, targets
            )

            # Perform inner loop adaptation
            local_model, output_weight, output_bias, classes = self.adapt_few_shot(
                support_inputs, support_targets
            )

            # Determine loss of query set
            query_labels = (
                (classes[None, :] == query_targets[:, None]).long().argmax(dim=-1)
            )
            loss, preds, acc = self.run_model(
                local_model, output_weight, output_bias, query_inputs, query_labels
            )

            # Calculate gradients for query set loss
            if mode == "train":
                loss.backward()
                # only update adapter fusion weights
                for (name_glob, p_glob), (name_local, p_local) in zip(self.model.named_parameters(), local_model.named_parameters()):
                    if p_glob.requires_grad and p_local.requires_grad:

                        # First-order approx. -> add gradients of finetuned and base model
                        p_glob.grad += p_local.grad

            accuracies.append(acc.mean().detach())
            losses.append(loss.detach())

        # Perform update of base model
        if mode == "train":
            opt = self.optimizers()
            opt.step()
            opt.zero_grad()

        self.log(f"{mode}_loss", sum(losses) / len(losses))
        self.log(f"{mode}_acc", sum(accuracies) / len(accuracies))

    def training_step(self, batch, batch_idx):
        print(f'TRAINING STEP: {batch_idx}')
        self.outer_loop(batch, mode="train")
        return None  # Returning None means we skip the default training optimizer steps by PyTorch Lightning

    # def validation_step(self, batch, batch_idx):
    #     # Validation requires to finetune a model, hence we need to enable gradients
    #     torch.set_grad_enabled(True)
    #     self.outer_loop(batch, mode="val")
    #     torch.set_grad_enabled(False)
