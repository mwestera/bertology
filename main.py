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


def compute_CAT():  # Another measure: MEOW -- mean energy of words?? DOG -- ?

    # Starting activations are one hot vector for each token
    activations = np.diag(np.ones(num_tokens))      # n_tokens × n_tokens
    # This will contain the sum of all activations, either across layers or only per layer
    sum_activations = np.zeros_like(activations)
    for layer in attn_for_layers:
        for head in layer:      # n_tokens × n_tokens
            activations_per_head = np.matmul(head, activations)
            # (i,j) = how much (activations coming ultimately from) token i influences token j
            if NORMALIZE:       # Normalize influence (across all tokens i) on each token j
                for j in range(0, len(activations_per_head)):
                    activations_per_head[:,j] = normalize(activations_per_head[:,j])
            sum_activations += activations_per_head
        if METHOD == "percolate":
            # for the next layer, use sum_activations as the next input activations, and reset the sum.
            activations = sum_activations
            # I checked: normalizing the activations (as a whole or per col) makes no difference.
            sum_activations = np.zeros_like(activations)
        elif METHOD == "sum":
            # for the next layer, keep the sum_activations, and reset the activations.
            activations = np.diag(np.ones(num_tokens))

    if METHOD == "sum":
        activations = sum_activations

    return activations

bert_version = 'bert-base-cased'    # TODO Why no case?

tokenizer = BertTokenizer.from_pretrained(bert_version)

METHOD = "percolate"  # "sum"
NORMALIZE = True
    # What about normalizing per layer, instead of per head? Does that make any sense? Yes, a bit.
    # However, since BERT has LAYER NORM in each attention head, outputs of all heads will have same mean/variance.
    # Does this mean that all heads will contribute same amount of information? Yes, roughly.


LAYERS = [0]
# LAYERS = [[0,1,2], [3,4,5], [6,7,8], [9,10,11]]
# LAYERS = [[0,1,2,3,4,5,6,7,8,9,10,11]]
layer_inds = [0,1,2,3,4,5,6,7,8,9,10,11]
LAYERS = [layer_inds[:i] for i in range(1,13)]
# LAYERS = [10,11]

# sentence_a = "Every farmer who owns a donkey beats it."
# sentence_b = "He is wearing a gray raincoat."

# I want the ability to (i) compare minimal pairs, (ii) sets of them -- only pairs though? For visualisation perhaps, but there can be more factors if just for the stats...
DATA = [
        # "The boy has a cat while the girl has a pigeon.",
        # "The boy has a cat while the girl has a pigeon."
        # "The boy has a cat. He likes to stroke it.",
        # "The boy has no cat. He likes to stroke it.",
        'The teacher wants every boy to like himself.',
        'The teacher wants every boy to like him.',
        # "I cannot find one of my ten marbles. It's probably under the couch.",
        # "I only found nine of my ten marbles. It's probably under the couch.",
        # '|0 Every farmer | who |1 owns a donkey |2 beats it.',
        # '|0 No farmer | who |1 owns a donkey |2 beats it.',
        # "Few of the children ate their ice-cream. They ate the apple flavor first.",
        # "Few of the children ate their ice-cream. They threw it around the room instead.",
        # "Few of the children ate their ice-cream. The others threw it around the room instead.",
    ]


def parse_data(data):

    sentences = []
    group_ids_per_sentence = []

    for s in data:
        group_to_token_ids = {}
        sentence = ""
        total_len = 1   # Take mandatory CLS symbol into account
        for each_part in s.strip('|').split('|'):
            first_char = each_part[0]
            if first_char.isdigit():
                group_id = int(first_char)
                each_part = each_part[1:].strip()
            tokens = tokenizer.tokenize(each_part)
            if first_char.isdigit():
                if group_id in group_to_token_ids:
                    group_to_token_ids[group_id].append(list(range(total_len, total_len + len(tokens))))
                else:
                    group_to_token_ids[group_id] = list(range(total_len, total_len + len(tokens)))
            total_len += len(tokens)
            sentence += each_part.strip() + ' '
        sentences.append(sentence.strip())
        group_ids_per_sentence.append(group_to_token_ids)

    return sentences, group_ids_per_sentence


print(parse_data(DATA))
# TODO Further implement using this; basically matters only in final plot, right?


# IDEA: Compare quantifiers, restrictor vs scope, to see how the info flows to the quantifier... advantage: very uniform sentences...

model = BertModel.from_pretrained(bert_version)
attention_visualizer = visualization.AttentionVisualizer(model, tokenizer)

# First, I want to compare pairs of sentences across layers. Only afterwards think about accumulating stats of multiple sentences.

AATs_per_layer = [[] for _ in LAYERS]

for sequence in DATA:

    tokens_a, tokens_b, attn = attention_visualizer.get_viz_data(sequence)
    all_tokens = tokens_a + tokens_b
    num_tokens = len(all_tokens)
    attn = attn.squeeze()

    # TODO Handle the cumulative case smarter (lot of redundancy: 1... 1,2... 1,2,3... 1,2,3,4...)
    # TODO Include cumulativity automatically on each run? Both methods (sum/percolate)?
    for layers_idx, layers in enumerate(LAYERS):
        if isinstance(layers, int):
            layers = [layers]

        attn_for_layers = attn[layers]

        AAT = compute_CAT() # TODO Rename this function and remove the old way.
        AAT = AAT.transpose()       # (i,j) --> (j,i): how much j is unfluenced by i
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
        ax = sns.heatmap(AAT, xticklabels=True, yticklabels=True, vmin=vmin, vmax=vmax, linewidth=0.5, ax=axs[i], cbar=False, cmap="Blues", square=True, cbar_kws={'shrink':.5}, label='small')
        ax.xaxis.tick_top()
        plt.setp(ax.get_xticklabels(), rotation=90)

    ## Difference plot
    diff = AATs[0] - AATs[1].values
    # TODO change values to be only shared tokens
    vmin, vmax = diff.min().min(), diff.max().max()
    vmin = -(max(0-vmin, vmax))
    vmax = (max(0-vmin, vmax))

    ax = sns.heatmap(diff, xticklabels=True, yticklabels=True, vmin=vmin, vmax=vmax, cbar=False, linewidth=0.5, ax=axs[i+1], cmap="coolwarm_r", square=True, cbar_kws={'shrink':.5}, label='small')
    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), rotation=90)

    pylab.savefig("output/temp{}.png".format('-'.join([str(i) for i in layers])))
    # pylab.show()


# import imageio
# images = []
# for filename in filenames:
#     images.append(imageio.imread(filename))
# imageio.mimsave('/path/to/movie.gif', images, format='GIF', duration=5)