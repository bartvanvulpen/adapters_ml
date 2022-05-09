from dataloader import load_mnli, load_sst2
import torch
import transformers
from transformers import BertTokenizer, EarlyStoppingCallback
from transformers import BertConfig, BertModelWithHeads
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from transformers.adapters.composition import Fuse
import torch.utils.data as data
from ml_batch_samplers import FewShotBatchSampler, TaskBatchSampler
from transformers import BertTokenizer

# def load_bert_model(id2label):
#     config = BertConfig.from_pretrained(
#         "bert-base-uncased",
#         id2label=id2label,
#     )
#     model = BertModelWithHeads.from_pretrained(
#         "bert-base-uncased",
#         config=config,
#     )
#     return model
#
# def setup_ada_fusion(model, id2label, target_task):
#     # Load the pre-trained adapters we want to fuse
#     model.load_adapter("nli/multinli@ukp", load_as="multinli", with_head=False, config='pfeiffer')
#     model.load_adapter("sts/qqp@ukp", load_as="qqp", with_head=False, config='pfeiffer')
#     model.load_adapter("sentiment/sst-2@ukp", load_as="sst", with_head=False, config='pfeiffer')
#     model.load_adapter("comsense/winogrande@ukp", load_as="wgrande", with_head=False, config='pfeiffer')
#     model.load_adapter("qa/boolq@ukp", load_as="boolq", with_head=False, config='pfeiffer')
#
#     # Add a fusion layer for all loaded adapters
#     adapter_setup = Fuse("multinli", "qqp", "sst", "wgrande", "boolq")
#     model.add_adapter_fusion(adapter_setup)
#     model.set_active_adapters(adapter_setup)
#
#     # Add a classification head for our target task
#     model.add_classification_head(f'{target_task}_classifier', num_labels=len(id2label))
#     model.train_adapter_fusion(adapter_setup)
#
#     return model


dataset, id2label = load_sst2()
print(id2label)

class NLPDataset(data.Dataset):

    def __init__(self, inputs, targets):
        super().__init__()
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, idx):
        input, target = self.inputs[idx], self.targets[idx]
        return input, target

    def __len__(self):
        return self.input.shape[0]



sst_train = NLPDataset(inputs=dataset['train']['input_ids'], targets=dataset['train']['labels'])
sst_val =  NLPDataset(inputs=dataset['validation']['input_ids'], targets=dataset['validation']['labels'])
sst_test =  NLPDataset(inputs=dataset['test']['input_ids'], targets=dataset['test']['labels'])

N_WAY = 2
K_SHOT = 4
train_loader = data.DataLoader(sst_train, batch_sampler=FewShotBatchSampler(sst_train.targets,
                                                                      include_query=True,
                                                                      N_way=N_WAY,
                                                                      K_shot=K_SHOT,
                                                                      shuffle=True), num_workers=0)
val_loader = data.DataLoader(sst_val, batch_sampler=FewShotBatchSampler(sst_val.targets,
                                                                      include_query=True,
                                                                      N_way=N_WAY,
                                                                      K_shot=K_SHOT,
                                                                      shuffle=False, shuffle_once=True), num_workers=0)

# Training set
train_protomaml_sampler = TaskBatchSampler(sst_train.targets,
                                           include_query=True,
                                           N_way=N_WAY,
                                           K_shot=K_SHOT,
                                           batch_size=16)

train_protomaml_loader = data.DataLoader(sst_train,
                                         batch_sampler=train_protomaml_sampler,
                                         collate_fn=train_protomaml_sampler.get_collate_fn(),
                                         num_workers=0)

for x in train_protomaml_loader:
    print(x[0][0].size(git))
    break

