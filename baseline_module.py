from transformers import BertAdapterModel
from transformers.adapters.composition import Fuse
import pytorch_lightning as pl
import torch
import warnings

warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()

# maps task name to the number of classes of each task
task_to_n_classes = {'mnli': 3, 'qqp': 2, 'sst': 2, 'wgrande': 2, 'imdb': 2, 'scitail': 2, 'argument': 3, 'boolq': 2, 'mrpc': 2, 'sick': 3, 'rte': 2, 'cb': 3}

# maps task name to the adapter identifier on adapter hub
task_to_adapter = {'mnli': 'nli/multinli@ukp', 'qqp': 'sts/qqp@ukp', 'sst': 'sentiment/sst-2@ukp', 'wgrande': 'comsense/winogrande@ukp', 'imdb': 'sentiment/imdb@ukp', 'scitail': 'nli/scitail@ukp', 'argument': 'argument/ukpsent@ukp', 'boolq': 'qa/boolq@ukp', 'mrpc': 'sts/mrpc@ukp', 'sick': 'nli/sick@ukp', 'rte': 'nli/rte@ukp', 'cb': 'nli/cb@ukp', 'hswag': 'comsense/hellaswag@ukp', 'siqa': 'comsense/siqa@ukp', 'csqa': 'comsense/csqa@ukp', 'cqa': 'comsense/cosmosqa@ukp'}


class BaselineModel(pl.LightningModule):
    def __init__(self, adapter_tasks, train_tasks, k, exp_num, lr=5e-5):
        """
        adapter_tasks - list of strings with the names of the tasks whose pre-trained adapters should be injected
        train_tasks - list of strings with the names of the tasks which
        will be used for training. This is needed to load the corresponding
        classification heads.
        all task names: ['mnli','qqp','sst','wgrande','imdb','scitail','argument','boolq','mrpc','sick','rte','cb'] + multiple choice tasks ['hswag', 'siqa', 'cqa', 'csqa']
        k - the k values determining the number of training examples per class
        exp_num - the experiment number this model is a baseline for, corresponding to the experiments sheet
        lr - learning rate
        """
        super().__init__()

        # save provided init arguments. Arguments k and exp_num (which are unused) 
        # are only provided to ensure they are saved as a hyperparameter
        self.save_hyperparameters()

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
        self.log('train_loss', total_loss)

        return total_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


    def prepare_for_test(self, task_name):
        """
        prepares this model for testing on a single task
        """
        self.model.add_classification_head('test', num_labels=task_to_n_classes[task_name])


    def train_step_single_task(self, batch):
        """
        This method should be used for the few-shot training step during testing.
        this method is analogous to training_step, with the difference
        that this method expects batch to contain data for a single task. 
        I.e. batch should be a dict with values 'input_ids', 'attention_ids', etc.
        """

        self.model.active_head = 'test'
        model_output = self.model.forward(**batch)
        return model_output.loss


    def compute_accuracy(self, batch):
        """
        Feeds forward the (single task) batch and returns the accuracy.
        Batch should contain a 'labels' key.
        """
        self.model.active_head = 'test'
        model_output = self.model.forward(**batch)

        predictions = torch.argmax(model_output.logits, dim=1)
        targets = torch.flatten(batch['labels'])

        accuracy = torch.sum(predictions == targets) / targets.size(0)

        return accuracy.item()
        
