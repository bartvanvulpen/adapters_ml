from responses import target
from transformers import BertAdapterModel
from transformers.adapters.composition import Fuse
import pytorch_lightning as pl
import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()


class BaselineModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        model = BertAdapterModel.from_pretrained('bert-base-uncased')

        adapter_kwargs = {'with_head': False, 'config': 'pfeiffer'}
        
        ds_to_n_classes = {'mnli': 3, 'qqp': 2, 'sst': 2, 'wgrande': 2, 'imdb': 2, 'scitail': 2, 'argument': 3, 'boolq': 2, 'mrpc': 2, 'sick': 3, 'rte': 2, 'cb': 3}

        train_datasets = ['imdb', 'mrpc', 'argument', 'scitail']

        model.load_adapter('nli/multinli@ukp', load_as='mnli', **adapter_kwargs)
        model.load_adapter('sentiment/sst-2@ukp', load_as='sst', **adapter_kwargs)
        model.load_adapter('sts/qqp@ukp', load_as='qqp', **adapter_kwargs)
        model.load_adapter('comsense/winogrande@ukp', load_as='wgrande', **adapter_kwargs)
        model.load_adapter('qa/boolq@ukp', load_as='boolq', **adapter_kwargs)
        
        for ds_name in train_datasets:
            model.add_classification_head(ds_name, num_labels=ds_to_n_classes[ds_name])

        adapter_setup = Fuse('mnli', 'qqp', 'sst', 'wgrande', 'boolq')
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
            model_output = self.model.forward(**batch[ds_name])
            # the model output contains the CrossEntropyLoss.
            # because labels were provided to the forward function
            # HuggingFaces already computes loss
            losses.append(model_output.loss)

        total_loss = sum(losses)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)



