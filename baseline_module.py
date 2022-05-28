from responses import target
from transformers import BertAdapterModel
from transformers.adapters.composition import Fuse
import pytorch_lightning as pl
import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()

task_to_n_classes = {'mnli': 3, 'qqp': 2, 'sst': 2, 'wgrande': 2, 'imdb': 2, 'scitail': 2, 'argument': 3, 'boolq': 2, 'mrpc': 2, 'sick': 3, 'rte': 2, 'cb': 3}

task_to_adapter = {'mnli': 'nli/multinli@ukp', 'qqp': 'sts/qqp@ukp', 'sst': 'sentiment/sst-2@ukp', 'wgrande': 'comsense/winogrande@ukp', 'imdb': 'sentiment/imdb@ukp', 'scitail': 'nli/scitail@ukp', 'argument': 'argument/ukpsent@ukp', 'boolq': 'qa/boolq@ukp', 'mrpc': 'sts/mrpc@ukp', 'sick': 'nli/sick@ukp', 'rte': 'nli/rte@ukp', 'cb': 'nli/cb@ukp'}


class BaselineModel(pl.LightningModule):
    def __init__(self, adapter_tasks, train_tasks, lr=1e-3):
        """
        adapter_tasks - list of strings with the names of the tasks whose pre-trained adapters should be injected
        train_tasks - list of strings with the names of the tasks which
        will be used for training. This is needed to load the corresponding
        classification heads.
        all task names: ['mnli','qqp','sst','wgrande','imdb','scitail','argument','boolq','mrpc','sick','rte','cb']
        lr - learning rate
        """
        super().__init__()

        self.lr = lr

        model = BertAdapterModel.from_pretrained('bert-base-uncased')

        # inject all adapters into the model
        for task_name in adapter_tasks:
            model.load_adapter(
                task_to_adapter[task_name], 
                load_as=task_name, 
                with_head=False,
                config='pfeiffer'
            )
        
        # add a classification head for each training task to the model
        for task_name in train_tasks:
            model.add_classification_head(task_name, num_labels=task_to_n_classes[task_name])

        # add an adapter fusion layer and freeze all weights except
        # for the fusion weights
        adapter_setup = Fuse(*adapter_tasks)
        model.add_adapter_fusion(adapter_setup)
        model.set_active_adapters(adapter_setup)
        model.train_fusion(adapter_setup)

        self.model = model


    def training_step(self, batch, batch_idx):
        """
        batch is a dict with the dataset names as keys, and a dict
        with all model inputs as values
        """
        losses = []

        for ds_name in batch:
            self.model.active_head = ds_name
            
            # batch[ds_name] is a dictionary with precisely the named
            # arguments that forward() requires. 
            model_output = self.model.forward(**batch[ds_name])

            # the model output contains the average CrossEntropyLoss.
            # because labels were provided to the forward function
            # HuggingFaces already computes loss
            losses.append(model_output.loss)

        # add the losses of all datasets
        total_loss = sum(losses)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
