from bertviz import attention, visualization
from bertviz.pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pylab as pylab

import pandas as pd

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


bert_version = 'bert-base-cased'    # TODO Why no case?

tokenizer = BertTokenizer.from_pretrained(bert_version)

NORMALIZE = True
# LAYERS = [[0,1,2], [3,4,5], [6,7,8], [9,10,11]]
# LAYERS = [[0,1,2,3,4,5,6,7,8,9,10,11]]
LAYERS = [0,1,2,3,4,5,6,7,8,9,10,11]
LAYERS = [LAYERS[:i] for i in range(1,13)]
# LAYERS = [10,11]

# sentence_a = "Every farmer who owns a donkey beats it."
# sentence_b = "He is wearing a gray raincoat."
DATA = [
        # "The boy has a cat while the girl has a pigeon.",
        # "The boy has a cat while the girl has a pigeon."
        # "The boy has a cat. He likes to stroke it.",
        # "The boy has no cat. He likes to stroke it.",
        "The teacher wants every boy to like himself.",
        "The teacher wants every boy to like him.",
        #("I've found nine out of ten marbles.", "It's probably under the couch."),
        #("I've found all marbles except for one.", "It's probably under the couch."),
        # ("Few of the children ate their ice-cream.", "They ate the apple flavor first."),
        # ("Few of the children ate their ice-cream.", "They threw it around the room instead."),
        # ("Few of the children ate their ice-cream.", "The others threw it around the room instead."),
    ]

model = BertModel.from_pretrained(bert_version)
attention_visualizer = visualization.AttentionVisualizer(model, tokenizer)

# First, I want to compare pairs of sentences across layers. Only afterwards think about accumulating stats of multiple sentences.

AATs_per_layer = [[] for _ in LAYERS]

for sequence in DATA:

    tokens_a, tokens_b, attn = attention_visualizer.get_viz_data(sequence)
    all_tokens = tokens_a + tokens_b
    num_tokens = len(all_tokens)
    attn = attn.squeeze()

    for layers_idx, layers in enumerate(LAYERS):

        if isinstance(layers, int):
            layers = [layers]

        attn_for_layers = attn[layers]

        AAT = []
        for token_idx, token in enumerate(all_tokens):      # TODO alternative would be to simply add up all weights... More ad hoc, but could be interesting.
            token_onehot = np.zeros(num_tokens)
            token_onehot[token_idx] = 1
            AA = None
            for layer in attn_for_layers:
                for head in layer:
                        attention = np.matmul(head, token_onehot)
                        if NORMALIZE:
                            attention = normalize(attention)
                        if AA is None:
                            AA = attention
                        else:
                            AA += attention
            AAT.append(AA)

        AAT = np.stack(AAT)
        AAT = pd.DataFrame(AAT, index=all_tokens, columns=all_tokens)

        AATs_per_layer[layers_idx].append(AAT)

for AATs, layers in zip(AATs_per_layer, LAYERS):
    if isinstance(layers, int):
        layers = [layers]

    fig, axs = plt.subplots(ncols=len(AATs) + 1, figsize=(12,4))
    plt.subplots_adjust(wspace = .6, top = .9)
    fig.suptitle("Layer "+','.join([str(i) for i in layers]))

    # Individual sequence plots
    vmin = min([AAT.min().min() for AAT in AATs])
    vmax = max([AAT.max().max() for AAT in AATs])

    for i, AAT in enumerate(AATs):
        ax = sns.heatmap(AAT, vmin=vmin, vmax=vmax, linewidth=0.5, ax=axs[i], cbar=False, cmap="Blues", square=True, cbar_kws={'shrink':.5})
        ax.xaxis.tick_top()
        plt.setp(ax.get_xticklabels(), rotation=90)

    ## Difference plot
    diff = AATs[0] - AATs[1].values
    vmin, vmax = diff.min().min(), diff.max().max()
    vmin = -(max(0-vmin, vmax))
    vmax = (max(0-vmin, vmax))

    ax = sns.heatmap(diff, vmin=vmin, vmax=vmax, cbar=False, linewidth=0.5, ax=axs[i+1], cmap="coolwarm_r", square=True, cbar_kws={'shrink':.5})
    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), rotation=90)

    pylab.savefig("output/temp{}.png".format('-'.join([str(i) for i in layers])))
    # pylab.show()
