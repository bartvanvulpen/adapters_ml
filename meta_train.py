## Standard libraries
import os
import argparse
import torch

## PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset_loader import ArgumentDatasetSplit

## Dataset and sampler
from data_construction import get_train_loader
from sampler import split_batch
from ProtoMAML import ProtoMAML

## Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "metatrain_outputs/"
if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)


def train_model(model_class, train_loader, max_n_steps, save_dir_name, debug=False, **kwargs):

    print('Debug:', debug)
    trainer = pl.Trainer(fast_dev_run=debug,
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_dir_name),
        gpus=1 if torch.cuda.is_available() else 0,
        max_steps=max_n_steps,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="train_loss", verbose=True, every_n_train_steps=5)],
        progress_bar_refresh_rate=0, log_every_n_steps=1,
    )
    trainer.logger._default_hp_metric = None

    # if pretrained model exists, use it
    pretrained_filename = os.path.join(CHECKPOINT_PATH, model_class.__name__ + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = model_class.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = model_class(**kwargs)

        trainer.fit(model, train_loader)

        if debug == False:
            model = model_class.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path
            )

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--tasks', nargs="+", default=[],
                        help='Training tasks to use for meta_training for', required=True)

    parser.add_argument('--inner_steps', type=int, default=5,
                        help='number of inner steps')

    parser.add_argument('--k_shot', type=int, default=4,
                        help='number of samples to take per class')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')

    parser.add_argument('--lr_inner', type=float, default=0.1,
                        help='inner learning rate')

    parser.add_argument('--lr_outer', type=float, default=0.1,
                        help='outer learning rate')

    parser.add_argument('--n_workers', type=int, default=0,
                        help='Number of workers')

    parser.add_argument('--debug', action='store_true',
                        help='Use --debug when you want to do a quick dev run')

    parser.add_argument('--max_steps', type=int, default=200,
                        help='Maximum number of steps to train on')

    parser.add_argument('--task_batch_size', type=int, default=16,
                        help='Number of times to sample a task in TaskBatchSampler')

    parser.add_argument('--adapters', nargs="+", default=[],
                        help='Adapter to use for meta learning', required=True)

    parser.add_argument('--adapterfusion_path', type=str,
                        help='Optional path to pretrained adapterfusion', required=False)


    args = parser.parse_args()

    # remove possible duplicates
    tasks = list(set(args.tasks))
    adapters = list(set(args.adapters))
    tasks.sort()
    adapters.sort()

    print('META-TRAINING TASKS:', tasks)
    print('ADAPTERS USED:', adapters)

    save_name = "_".join(adapters) + '-' + "_".join(tasks)

    train_loader = get_train_loader(tasks, K_SHOT=args.k_shot, task_batch_size=args.task_batch_size, num_workers=args.n_workers)

    print("Starting training...")
    trained_protomaml_model = train_model(
        ProtoMAML,
        lr=args.lr,
        lr_inner=args.lr_inner,
        lr_output=args.lr_outer,
        num_inner_steps=args.inner_steps,
        adapters_used=adapters,
        k_shot=args.k_shot,
        task_batch_size=args.task_batch_size,
        tasks=tasks,
        train_loader=train_loader,
        max_n_steps=args.max_steps,
        save_dir_name=save_name,
        debug=args.debug,
        adapterfusion_path=args.adapterfusion_path
    )
