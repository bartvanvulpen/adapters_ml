from transformers.adapters import BertAdapterModel
from transformers.adapters.composition import Fuse
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


class IntBert(BertAdapterModel):
    def __init__(self, *args):
        super().__init__(*args)

    def get_adapters(self, adapters, adapterfusion_path=None):
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
        if adapterfusion_path:
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
    layer, datasets, dataset_names, adapters, adapterfusion_paths
):
    # layer is 1, ..., 12
    # example use: create_softmax_visual(2, [dataset1[0][val_key1],
    #                       dataset2[0][val_key2],
    #                       dataset3[0][val_key3],
    #                       dataset4[0][val_key4]], ["multinli","qqp", "sst","boolq"], {
    #                       "nli/multinli@ukp": "multinli",
    #                       "sts/qqp@ukp": "qqp",
    #                       "sentiment/sst-2@ukp": "sst",
    #                       "comsense/winogrande@ukp": "wgrande",
    #                       "qa/boolq@ukp": "boolq",
    #                     }, ["fusion_weights/mnli", "fusion_weights/qqp", "fusion_weights/sst", "fusion_weights/boolq"])
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = IntBert.from_pretrained("bert-base-uncased")
    model.eval()
    vis = []
    with torch.no_grad():
        for i, dataset in enumerate(datasets):
            try:
                model.get_adapters(adapters, adapterfusion_paths[i])
            except:
                model.get_adapters(adapters, None)
            model = model.to(device)
            embedding, first_layers, rest_layers, adapts, q, k = model.fix_layers(
                layer - 1
            )
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=128)
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
    sns.heatmap(
        torch.vstack(vis),
        xticklabels=list(adapters.values()),
        yticklabels=dataset_names,
        linewidths=0.10,
        annot=True,
    )
    plt.title("AdapterFusion activations of pretrained ST-Adapters")
    plt.show()
