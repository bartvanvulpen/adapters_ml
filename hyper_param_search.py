#   https://towardsdatascience.com/how-to-tune-pytorch-lightning-hyperparameters-80089a281646


## Standard libraries
import os
import argparse
import torch
from torch.nn import functional as F

## PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset_loader import ArgumentDatasetSplit

## Ray tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune

## Dataset and sampler
from data_construction import get_train_val_loaders
from sampler import split_batch
from ProtoMAML import ProtoMAML

## Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "checkpoints_hyperparam_search/"
os.environ["SLURM_JOB_NAME"] = "bash"

def train_model(config, model_class, train_loader, val_loader):
    debug = False
    metrics = {"loss": "val_loss", "acc": "val_acc"}
    trainer = pl.Trainer(fast_dev_run=debug,
        default_root_dir=os.path.join(CHECKPOINT_PATH, model_class.__name__),
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=50,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
            TuneReportCallback(metrics, on="validation_end")
        ],
        progress_bar_refresh_rate=0,
    )

    model = model_class(**config)
    trainer.fit(model, train_loader, val_loader)

def tune_run(model_class, proto_dim, train_loader, val_loader, args):
    config = {
        "proto_dim": 768,
        "lr": tune.loguniform(1e-4, 1e-1),
        "lr_inner": tune.loguniform(1e-4, 1e-1),
        "lr_output": tune.loguniform(1e-4, 1e-1),
        "num_inner_steps": tune.choice([1,3,5,7,10,15,20])
    }
    trainable = tune.with_parameters(
        train_model,
        model_class=model_class,
        train_loader=train_loader,
        val_loader=val_loader
    )

    analysis = tune.run(
        trainable,
        resources_per_trial={
            "cpu": 1,
            "gpu": args.gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=args.num_samples,
        name="tune_metafusion",
        max_concurrent_trials=1,
    )

    print(analysis.best_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_tasks', nargs="+", default=['imdb', 'mrpc', 'argument', 'scitail'],
                        help='training task to use for meta_training for')

    parser.add_argument('--val_tasks', nargs="+", default=['sick', 'rte', 'cb'],
                        help='validation task to use for meta_training for')

    parser.add_argument('--gpus_per_trial', type=int, default=1,
                        help='amount of gpus used for each trial')

    parser.add_argument('--num_samples', type=int, default=10,
                        help='total amount of hyperparam samples')

    parser.add_argument('--n_workers', type=int, default=0,
                        help='amount of workers for dataloader')

    args = parser.parse_args()

    train_loader, val_loader = get_train_val_loaders(args.train_tasks, args.val_tasks, num_workers=args.n_workers)

    print("Starting hyperparam search...")
    pl.seed_everything(42)  # To be reproducable
    protomaml_model = tune_run(
        ProtoMAML,
        proto_dim=768,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args
    )
