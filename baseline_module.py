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

        model.load_adapter('nli/multinli@ukp', load_as='mnli', **adapter_kwargs)
        model.load_adapter('sentiment/sst-2@ukp', load_as='sst', **adapter_kwargs)
        model.load_adapter('sts/qqp@ukp', load_as='qqp', **adapter_kwargs)
        model.load_adapter('comsense/winogrande@ukp', load_as='wgrande', **adapter_kwargs)
        model.load_adapter('qa/boolq@ukp', load_as='boolq', **adapter_kwargs)

        model.add_classification_head('imdb', num_labels=)
        model.add_classification_head('mrpc', num_labels=)
        model.add_classification_head('argument', num_labels=)
        model.add_classification_head('scitail', num_labels=)

        adapter_setup = Fuse('mnli', 'qqp', 'sst', 'wgrande', 'boolq')
        model.add_adapter_fusion(adapter_setup)
        model.set_active_adapters(adapter_setup)
        model.train_fusion(adapter_setup)

        self.model = model


    def training_step(self, batch, batch_idx):
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)



