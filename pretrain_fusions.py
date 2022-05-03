from transformers import AutoTokenizer, AutoAdapterModel
from transformers import list_adapters
import dataloader
from adapter_fusion import load_bert_model
from transformers import BertTokenizer, EarlyStoppingCallback
from transformers import BertConfig, BertModelWithHeads
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from transformers.adapters.composition import Fuse
import numpy as np
import pytorch_lightning as pl


def setup_ada_fusion(model, id2label, target_task):
    # Load the pre-trained adapters we want to fuse
    model.load_adapter("nli/multinli@ukp", load_as="multinli", with_head=False, config='pfeiffer')
    model.load_adapter("sts/qqp@ukp", load_as="qqp", with_head=False, config='pfeiffer')
    model.load_adapter("sentiment/sst-2@ukp", load_as="sst", with_head=False, config='pfeiffer')
    model.load_adapter("comsense/winogrande@ukp", load_as="wgrande", with_head=False, config='pfeiffer')
    model.load_adapter("qa/boolq@ukp", load_as="boolq", with_head=False, config='pfeiffer')

    # Add a fusion layer for all loaded adapters
    adapter_setup = Fuse("multinli", "qqp", "sst", "wgrande", "boolq")
    model.add_adapter_fusion(adapter_setup)
    model.set_active_adapters(adapter_setup)

    # Add a classification head for our target task
    model.add_classification_head(f'{target_task}_classifier', num_labels=len(id2label))
    model.train_adapter_fusion(adapter_setup)

    return model


def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}

def train(model, training_args, dataset):
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_accuracy,
    )
    trainer.train()
    trainer.evaluate()


if __name__ == '__main__':
    target_task = "boolq"

    if target_task == 'boolq':
        dataset, id2label = dataloader.load_boolq()
    elif target_task == 'sst':
        dataset, id2label = dataloader.load_sst2()


    model = load_bert_model(id2label)


    model = setup_ada_fusion(model, id2label, target_task)
    # specify training args
    training_args = TrainingArguments(
        learning_rate=5e-5,
        num_train_epochs=10,
        per_device_train_batch_size=8,  # higher has memory problems on lisa
        per_device_eval_batch_size=8,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        lr_scheduler_type="constant",
        output_dir=f"training_output/{target_task}/checkpoints",
        logging_dir=f"training_output/{target_task}/logs",
        overwrite_output_dir=True,
        do_predict=True,
        # load_best_model_at_end=True, <- this does not work currently, probably a bug from AdapterHub
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
    )

    train(model, training_args, dataset)
    model.save_adapter_fusion("./saved/fusion/boolq_run", "multinli,qqp,sst,wgrande,boolq")
    model.save_all_adapters("./saved/sep_adapters/boolq_run")





