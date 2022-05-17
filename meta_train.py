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
from dataset_loader import ArgumentDatasetSplit

## Dataset and sampler
from data_construction import get_train_val_loaders
from sampler import split_batch
from ProtoMAML import ProtoMAML

## Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "checkpoints_meta_learning/"

def train_model(model_class, train_loader, val_loader, **kwargs):

    debug = True
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_tasks', nargs="+", default=['sst', 'boolq'],
                        help='task to use for meta_training for')

    parser.add_argument('--val_tasks', nargs="+", default=['mnli', 'qqp'],
                        help='task to use for meta_training for')

    parser.add_argument('--inner_steps', type=int, default=1,
                        help='number of inner steps')

    parser.add_argument('--k_shot', type=int, default=4,
                        help='number of samples to take per class')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')

    parser.add_argument('--lr_inner', type=float, default=0.1,
                        help='learning rate')

    parser.add_argument('--lr_outer', type=float, default=0.1,
                        help='learning rate')

    parser.add_argument('--n_workers', type=int, default=0,
                        help='learning rate')



    args = parser.parse_args()

    train_loader, val_loader = get_train_val_loaders(args.train_tasks, args.val_tasks, num_workers=args.n_workers)

    print("Starting training...")
    protomaml_model = train_model(
        ProtoMAML,
        proto_dim=768,
        lr=args.lr,
        lr_inner=args.lr_inner,
        lr_output=args.lr_outer,
        num_inner_steps=args.inner_steps,  # Often values between 1 and 10
        train_loader=train_loader,
        val_loader=val_loader,
    )

