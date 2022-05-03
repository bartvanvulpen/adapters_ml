import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from transformers.adapters import BertAdapterModel

model = BertAdapterModel.from_pretrained("bert-base-uncased")
model.load_adapter("nli/multinli@ukp", load_as="multinli", with_head=False)
model.load_adapter("sts/qqp@ukp", with_head=False)
model.load_adapter("sentiment/sst-2@ukp", with_head=False)
model.load_adapter("comsense/winogrande@ukp", with_head=False)
model.load_adapter("sentiment/imdb@ukp", with_head=False)
model.load_adapter("comsense/hellaswag@ukp", with_head=False)
model.load_adapter("comsense/siqa@ukp", with_head=False)
model.load_adapter("comsense/cosmosqa@ukp", with_head=False)
model.load_adapter("nli/scitail@ukp", with_head=False)
model.load_adapter("argument/ukpsent@ukp", with_head=False)
model.load_adapter("comsense/csqa@ukp", with_head=False)
model.load_adapter("qa/boolq@ukp", with_head=False)
model.load_adapter("sts/mrpc@ukp", with_head=False)
model.load_adapter("nli/sick@ukp", with_head=False)
model.load_adapter("nli/rte@ukp", with_head=False)
model.load_adapter("nli/cb@ukp", with_head=False)


def get_params(adapter, layer):
    weight_up = model.state_dict()[
        f"bert.encoder.layer.{layer}.output.adapters.{adapter}.adapter_up.weight"
    ]
    weight_down = model.state_dict()[
        f"bert.encoder.layer.{layer}.output.adapters.{adapter}.adapter_down.0.weight"
    ]
    params = torch.vstack([weight_up, weight_down.T])
    return params


def get_cosine_sim(layer=None):
    """Plots cosine similarity heatmap for adapter weights. Weights for both
    parts of the adapter are concatenated and biases are excluded. If a layer is
    specified, only use adapter weights from that layer, otherwise concatenate
    weights from all layers"""

    adapters = [
        "multinli",
        "qqp",
        "sst-2",
        "winogrande",
        "imdb",
        "hellaswag",
        "socialiqa",
        "cosmosqa",
        "scitail",
        "argument",
        "csqa",
        "boolq",
        "mrpc",
        "sick",
        "rte",
        "cb",
    ]
    params = {}
    for a in adapters:
        if layer:
            params[a] = get_params(a, layer)
        else:
            params[a] = get_params(a, 0)
            for l in range(1, 12):
                params[a] = torch.vstack([params[a], get_params(a, l)])
    df_sims = pd.DataFrame(index=adapters, columns=adapters)
    for i, a in enumerate(params.values()):
        for j, b in enumerate(params.values()):
            if i == j:
                df_sims.loc[adapters[i], adapters[j]] = 0
            else:
                df_sims.loc[adapters[i], adapters[j]] = abs(
                    F.cosine_similarity(a, b).mean().item()
                )
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(df_sims.astype("float"), cmap="RdYlGn", linewidths=0.30, annot=True)
    plt.title("Cosine similarity of adapter weight matrices")
    plt.show()


get_cosine_sim()
