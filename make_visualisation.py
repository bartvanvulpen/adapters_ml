from transformers.adapters import BertAdapterModel
from transformers.adapters.composition import Fuse
from transformers import BertConfig
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from ProtoMAML import ProtoMAML
from dataset_loader import load_dataset_from_file, ArgumentDatasetSplit
import pickle as pkl

class IntBert(BertAdapterModel):
    def __init__(self, *args):
        super().__init__(*args)

    def get_adapters(self, adapters, adapterfusion_path=None, meta_tested=False):
        self.adas = adapters
        for a in self.adas:
            self.load_adapter(
                a, load_as=self.adas[a], with_head=False, config="pfeiffer"
            )
        adapter_setup = Fuse(*self.adas.values())
        try:
            self.delete_adapter_fusion(",".join(self.adas.values()))
        except:
            pass
        if meta_tested and adapterfusion_path:
            self.load_state_dict(torch.load(adapterfusion_path, map_location=torch.device('cpu')))
        elif adapterfusion_path:
            self.load_adapter_fusion(
                adapterfusion_path,
                load_as=",".join(self.adas.values()),
                set_active=True,
            )
        else:
            self.add_adapter_fusion(adapter_setup)
            self.set_active_adapters(adapter_setup)

    def fix_layers(self, output_layer):
        first_layers = []
        rest_layers = []
        adapts = []
        for i, l in self.iter_layers():
            if i < output_layer:
                first_layers.append(l)
        for n, m in self.named_modules():
            if n == "bert.embeddings":
                embedding = m
            if n == f"bert.encoder.layer.{output_layer}.attention":
                first_layers.append(m)
            if n == f"bert.encoder.layer.{output_layer}.intermediate":
                first_layers.append(m)
            if n == f"bert.encoder.layer.{output_layer}.output.dense":
                first_layers.append(m)
            if n == f"bert.encoder.layer.{output_layer}.output.LayerNorm":
                rest_layers.append(m)
            if n == f"bert.encoder.layer.{output_layer}.output.dropout":
                rest_layers.append(m)
            if n in [
                f"bert.encoder.layer.{output_layer}.output.adapters.{self.adas[a]}"
                for a in self.adas
            ]:
                adapts.append(m)
            if (
                n
                == f"bert.encoder.layer.{output_layer}.output.adapter_fusion_layer.{','.join(self.adas.values())}.query"
            ):
                q = m
            if (
                n
                == f"bert.encoder.layer.{output_layer}.output.adapter_fusion_layer.{','.join(self.adas.values())}.key"
            ):
                k = m
        return embedding, first_layers, rest_layers, adapts, q, k

    def _forward_impl(
        self, iid, tti, am, embedding, first_layers, rest_layers, adapts, q, k
    ):
        x = embedding(iid, tti, am)
        for l in first_layers:
            x = l(x)
            if type(x) == tuple:
                x = x[0]
        res_inp = x.detach().clone()
        for l in rest_layers:
            x = l(x)
        hq = q(res_inp)
        return F.softmax(
            torch.vstack(
                [
                    torch.tensordot(
                        hq, k(a(x, res_inp)[0]), dims=[(1, 2), (1, 2)]
                    ).diagonal()
                    for a in adapts
                ]
            ).T,
            dim=1,
        )

    def forward(self, iid, tti, am, embedding, first_layers, rest_layers, adapts, q, k):
        return self._forward_impl(
            iid, tti, am, embedding, first_layers, rest_layers, adapts, q, k
        )


def create_softmax_visual(
        layer, datasets, dataset_names, adapters, adapterfusion_paths, meta_tested=False
):
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = IntBert.from_pretrained("bert-base-uncased")
    model.eval()
    vis = []
    with torch.no_grad():
        for i, dataset in enumerate(datasets):
            try:
                model.get_adapters(adapters, adapterfusion_paths[i], meta_tested=meta_tested)
            except:
                model.get_adapters(adapters, None)
            model = model.to(device)
            embedding, first_layers, rest_layers, adapts, q, k = model.fix_layers(
                layer - 1
            )
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
            try:
                vis.append(
                    torch.vstack(
                        [
                            model(
                                torch.vstack(batch["input_ids"]).to(device),
                                torch.vstack(batch["token_type_ids"]).to(device),
                                torch.vstack(batch["attention_mask"]).to(device),
                                embedding,
                                first_layers,
                                rest_layers,
                                adapts,
                                q,
                                k,
                            )
                        for batch in tqdm(dataloader)
                        ]
                    )
                    .mean(dim=0)
                    .cpu()
                )
            except:
                                vis.append(
                    torch.vstack(
                        [
                            model(
                                batch["input_ids"].to(device),
                                batch["token_type_ids"].to(device),
                                batch["attention_mask"].to(device),
                                embedding,
                                first_layers,
                                rest_layers,
                                adapts,
                                q,
                                k,
                            )
                        for batch in tqdm(dataloader)
                        ]
                    )
                    .mean(dim=0)
                    .cpu()
                )
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.set(font_scale=1.6)
    hm = sns.heatmap(
        torch.vstack(vis),
        linewidths=0.10,
        annot=True,
    )
    hm.set_xticklabels(list(adapters.values()), fontsize = 20)
    hm.set_yticklabels(dataset_names, fontsize=20)
    #plt.title("AdapterFusion activations of pretrained ST-Adapters")
    plt.show()

def save_adapterfusion(ckpt_path, path, adapters):
    model = ProtoMAML.load_from_checkpoint(ckpt_path)
    model.model.save_adapter_fusion(path, adapters)
    return f"{path}/pytorch_model_adapter_fusion.bin"

# create visualisation of meta-tested AdapterFusion module
layer = 12
adapters = {"qa/boolq@ukp": "boolq", "nli/multinli@ukp": "mnli", "sts/qqp@ukp": "qqp", "sentiment/sst-2@ukp": "sst", "comsense/winogrande@ukp": "wgrande"}
dataset_names = ["cb", "rte", "sick"]
k = 8
datasets = [load_dataset_from_file(ds_name)[0]["validation"] for ds_name in dataset_names]
adapterfusion_path_rte = f"save_state_dict/make_visualizations_adapt_{k}_rte.pt"
adapterfusion_path_cb = f"save_state_dict/make_visualizations_adapt_{k}_cb.pt"
adapterfusion_path_sick = f"save_state_dict/make_visualizations_adapt_{k}_sick.pt"
adapterfusion_paths = [adapterfusion_path_cb, adapterfusion_path_rte, adapterfusion_path_sick]
create_softmax_visual(
     layer, datasets, dataset_names, adapters, adapterfusion_paths, meta_tested=True
)

# create visualisation of AdapterFusion module right after meta-training
ckpt_path = "metatrain_outputs/second_hyp/boolq_mnli_qqp_sst_wgrande-argument_imdb_mrpc_scitail/lightning_logs/version_9183201/checkpoints/epoch=0-step=159.ckpt"
layer = 12
adapters = {"qa/boolq@ukp": "boolq", "nli/multinli@ukp": "mnli", "sts/qqp@ukp": "qqp", "sentiment/sst-2@ukp": "sst", "comsense/winogrande@ukp": "wgrande"}
dataset_names = ["cb", "rte", "sick"]
adapts = "boolq,mnli,qqp,sst,wgrande"
adapterfusion_path = save_adapterfusion(ckpt_path, "saved_adapter_fusions", adapts)
adapterfusion_paths = [adapterfusion_path] * len(dataset_names)
create_softmax_visual(layer, datasets, dataset_names, adapters, adapterfusion_paths, meta_tested=False)
