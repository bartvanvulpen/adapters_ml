## Standard libraries
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
from data_construction import get_train_val_loaders
from sampler import split_batch
from ProtoMAML import ProtoMAML


"""
TRAINING AND TESTING FUNCTIONS
"""

## Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "checkpoints_meta_learning/"

def train_model(model_class, train_loader, val_loader, **kwargs):

    debug = False
    trainer = pl.Trainer(fast_dev_run=debug,
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

        if debug == False:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_tasks', nargs="+", default=['sst', 'boolq'],
                        help='task to use for meta_training for')

    parser.add_argument('--val_tasks', nargs="+", default=['mnli', 'qqp'],
                        help='task to use for meta_training for')

    args = parser.parse_args()

    train_loader, val_loader = get_train_val_loaders(args.train_tasks, args.val_tasks, num_workers=0)

    print("Starting training...")
    protomaml_model = train_model(
        ProtoMAML,
        proto_dim=64,
        lr=1e-3,
        lr_inner=0.1,
        lr_output=0.1,
        num_inner_steps=1,  # Often values between 1 and 10
        train_loader=train_loader,
        val_loader=val_loader,
    )

# """
# TEST THE MODEL
# """
## TODO monday
# protomaml_result_file = os.path.join(CHECKPOINT_PATH, "protomaml_fewshot.json")
#
# if os.path.isfile(protomaml_result_file):
#     # Load pre-computed results
#     with open(protomaml_result_file, "r") as f:
#         protomaml_accuracies = json.load(f)
#     protomaml_accuracies = {int(k): v for k, v in protomaml_accuracies.items()}
# else:
#     # Perform experiments
#     protomaml_accuracies = dict()
#     for k in [2, 4, 8, 16, 32]:
#         protomaml_accuracies[k] = test_protomaml(protomaml_model, test_set, k_shot=k)
#     # Export results
#     with open(protomaml_result_file, "w") as f:
#         json.dump(protomaml_accuracies, f, indent=4)
#
# for k in protomaml_accuracies:
#     print(
#         f"Accuracy for k={k}: {100.0*protomaml_accuracies[k][0]:4.2f}% (+-{100.0*protomaml_accuracies[k][1]:4.2f}%)"
#     )
