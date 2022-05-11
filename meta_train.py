## Standard libraries
import os
from copy import deepcopy
from statistics import mean, stdev
import json

## PyTorch
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

## PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

## Dataset and sampler
from sampler import (
    TaskBatchSampler,
    dataset_from_tasks,
    DATASETS,
    TASK_IDS,
    FewShotBatchSampler,
    split_batch,
    DATALOADERS,
)

## Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../saved_models/tutorial16"


"""
Mock model for testing purposes --> Replace with adaptor fusion model
"""
from transformers import BertForSequenceClassification


def get_transformer_model(output_size):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=output_size
    )
    for param in model.bert.parameters():
        param.requires_grad = False
    return model


"""
Bert with adapter fusion
"""


def get_adapter_fusion_model(
    output_size,
    adapters={
        "nli/multinli@ukp": "multinli",
        "sts/qqp@ukp": "qqp",
        "sentiment/sst-2@ukp": "sst",
        "comsense/winogrande@ukp": "wgrande",
        "qa/boolq@ukp": "boolq",
    },
):
    model = BertAdapterModel.from_pretrained("bert-base-uncased")
    for a in adapters:
        model.load_adapter(a, load_as=adapters[a], with_head=False, config="pfeiffer")
    adapter_setup = Fuse(*adapters.values())
    model.add_adapter_fusion(adapter_setup)
    model.set_active_adapters(adapter_setup)
    # linear layer?
    return model


"""
PROTOMAML MODEL (incl function to calculate prototypes)
"""


class ProtoNet:
    @staticmethod
    def calculate_prototypes(features, targets):
        # Given a stack of features vectors and labels, return class prototypes
        # features - shape [N, proto_dim], targets - shape [N]
        classes, _ = torch.unique(targets).sort()  # Determine which classes we have
        prototypes = []
        for c in classes:
            p = features[torch.where(targets == c)[0]].mean(
                dim=0
            )  # Average class feature vectors
            prototypes.append(p)
        prototypes = torch.stack(prototypes, dim=0)
        # Return the 'classes' tensor to know which prototype belongs to which class
        return prototypes, classes


class ProtoMAML(pl.LightningModule):
    def __init__(self, proto_dim, lr, lr_inner, lr_output, num_inner_steps):
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
        self.model = get_transformer_model(output_size=self.hparams.proto_dim)
        # (ds, id2label), key = DATALOADERS['sst']
        #
        # model = load_bert_model(id2label)
        # self.model = setup_ada_fusion(model)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[140, 180], gamma=0.1
        )
        return [optimizer], [scheduler]

    def run_model(self, local_model, output_weight, output_bias, inputs, labels):

        # Execute a model with given output layer weights and inputs
        feats = local_model(inputs)
        preds = F.linear(feats, output_weight, output_bias)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float()

        return loss, preds, acc

    def adapt_few_shot(self, support_inputs, support_targets):

        # Determine prototype initialization
        support_feats = self.model(support_inputs)
        prototypes, classes = ProtoNet.calculate_prototypes(
            support_feats, support_targets
        )
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
        for _ in range(self.hparams.num_inner_steps):

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

        inputs, targets = batch
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

            for p_global, p_local in zip(
                self.model.parameters(), local_model.parameters()
            ):

                # First-order approx. -> add gradients of finetuned and base model
                p_global.grad += p_local.grad

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
        self.outer_loop(batch, mode="train")
        return None  # Returning None means we skip the default training optimizer steps by PyTorch Lightning

    def validation_step(self, batch, batch_idx):
        # Validation requires to finetune a model, hence we need to enable gradients
        torch.set_grad_enabled(True)
        self.outer_loop(batch, mode="val")
        torch.set_grad_enabled(False)


"""
TRAINING AND TESTING FUNCTIONS
"""


def train_model(model_class, train_loader, val_loader, **kwargs):

    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, model_class.__name__),
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=200,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
        ],
        progress_bar_refresh_rate=0,
    )
    trainer.logger._default_hp_metric = None

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, model_class.__name__ + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = model_class.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = model_class(**kwargs)
        trainer.fit(model, train_loader, val_loader)
        # Load best checkpoint after training
        model = model_class.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

    return model


def test_protomaml(model, dataset, k_shot=4):

    pl.seed_everything(42)
    model = model.to(device)
    num_classes = dataset.targets.unique().shape[0]
    exmps_per_class = dataset.targets.shape[0] // num_classes

    # Data loader for full test set as query set
    full_dataloader = data.DataLoader(
        dataset, batch_size=128, num_workers=4, shuffle=False, drop_last=False
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


# Training constant
N_WAY = 3  # All tasks have 2 or 3 classes, so set to 3 to ensure all classes are covered in an episode
K_SHOT = 4

"""
CONSTRUCT DATASETS
"""

combined_dataset = {
    "tasks": torch.hstack(
        [
            torch.tensor([TASK_IDS[ds_name]] * len(ds["labels"]))
            for ds_name, ds in DATASETS.items()
        ]
    ),
    "input_ids": torch.vstack([ds["input_ids"] for ds in DATASETS.values()]),
    "token_type_ids": torch.vstack([ds["token_type_ids"] for ds in DATASETS.values()]),
    "attention_mask": torch.vstack([ds["attention_mask"] for ds in DATASETS.values()]),
    "labels": torch.hstack([ds["labels"] for ds in DATASETS.values()]),
}

train_datasets = ["mnli", "sst"]
val_datasets = ["mnli", "qqp"]
test_datasets = ["cb"]

train_set = dataset_from_tasks(
    combined_dataset, torch.tensor([TASK_IDS[ds] for ds in train_datasets])
)
val_set = dataset_from_tasks(
    combined_dataset, torch.tensor([TASK_IDS[ds] for ds in val_datasets])
)
test_set = dataset_from_tasks(
    combined_dataset, torch.tensor([TASK_IDS[ds] for ds in test_datasets])
)


# Training set
train_protomaml_sampler = TaskBatchSampler(
    train_set.tasks,
    train_set.labels,
    include_query=True,
    N_way=N_WAY,
    K_shot=K_SHOT,
    batch_size=16,
    shuffle=True,  # Set to False, otherwise you risk getting same class twice in dataset
)
train_protomaml_loader = data.DataLoader(
    train_set,
    batch_sampler=train_protomaml_sampler,
    collate_fn=train_protomaml_sampler.get_collate_fn(),
    num_workers=0,
)
x = next(iter(train_protomaml_loader))
print(x)
# s, q, st, qt = split_batch(x, y)
# print(s)
# print(q)
# print(st)
# print(qt)


# Validation set
val_protomaml_sampler = TaskBatchSampler(
    val_set.tasks,
    val_set.labels,
    include_query=True,
    N_way=N_WAY,
    K_shot=K_SHOT,
    batch_size=1,  # We do not update the parameters, hence the batch size is irrelevant here
    shuffle=False,
)
val_protomaml_loader = data.DataLoader(
    val_set,
    batch_sampler=val_protomaml_sampler,
    collate_fn=val_protomaml_sampler.get_collate_fn(),
    num_workers=0,
)

"""
TRAIN THE MODEL !
"""
from adapter_fusion import load_bert_model
from transformers.adapters.composition import Fuse


def setup_ada_fusion(model):
    # Load the pre-trained adapters we want to fuse
    model.load_adapter(
        "nli/multinli@ukp", load_as="multinli", with_head=False, config="pfeiffer"
    )
    model.load_adapter("sts/qqp@ukp", load_as="qqp", with_head=False, config="pfeiffer")
    model.load_adapter(
        "sentiment/sst-2@ukp", load_as="sst", with_head=False, config="pfeiffer"
    )
    model.load_adapter(
        "comsense/winogrande@ukp", load_as="wgrande", with_head=False, config="pfeiffer"
    )
    model.load_adapter(
        "qa/boolq@ukp", load_as="boolq", with_head=False, config="pfeiffer"
    )

    # Add a fusion layer for all loaded adapters
    adapter_setup = Fuse("multinli", "qqp", "sst", "wgrande", "boolq")
    model.add_adapter_fusion(adapter_setup)
    model.set_active_adapters(adapter_setup)

    # Add a classification head for our target task
    # model.add_classification_head(f'{target_task}_classifier', num_labels=len(id2label))
    model.train_adapter_fusion(adapter_setup)

    return model


print("Starting training...")
protomaml_model = train_model(
    ProtoMAML,
    proto_dim=64,
    lr=1e-3,
    lr_inner=0.1,
    lr_output=0.1,
    num_inner_steps=1,  # Often values between 1 and 10
    train_loader=train_protomaml_loader,
    val_loader=val_protomaml_loader,
)

"""
TEST THE MODEL
"""
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
