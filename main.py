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


def compute_PAT(heads_per_layer, layer_norm=True):
    """
    Computes Percolated Attention per Token (PAT), through all layers.
    :param heads_per_layer:
    :param layer_norm:
    :return: percolated activations up to every layer
    """
    # Method 1: Percolating activations: Starting activations are one hot vector for each token
    percolated_activations_per_layer = []
    percolated_activations = np.diag(np.ones(heads_per_layer.shape[-1]))      # n_tokens × n_tokens
    for layer in heads_per_layer:
        summed_activations = np.zeros_like(percolated_activations)
        for head in layer:      # n_tokens × n_tokens
            activations_per_head = np.matmul(head, percolated_activations)
            # (i,j) = how much (activations coming ultimately from) token i influences token j
            if layer_norm:       # Normalize influence (across all tokens i) on each token j
                for j in range(0, len(activations_per_head)):
                    activations_per_head[:,j] = normalize(activations_per_head[:, j])
            summed_activations += activations_per_head
        # TODO store sum_activations
        # for the next layer, use sum_activations as the next input activations, and reset the sum.
        percolated_activations = summed_activations
        # I checked: normalizing the activations (as a whole or per col) makes no difference.

        percolated_activations_per_layer.append(percolated_activations)

    return percolated_activations_per_layer


def compute_MAT(heads_per_layer, layer_norm=True):
    """
    Computes Mean Attention per Token (MAT), i.e,, mean across all layers and heads.
    :param heads_per_layer:
    :param layer_norm:
    :return: Averages (across heads) per layer
    """
    # Method 2: summing activations
    mean_activations_per_layer = []
    summed_mean_activations = np.zeros_like(heads_per_layer[0][1])
    for heads_of_layer in heads_per_layer:
        summed_activations = np.zeros_like(heads_per_layer[0][1])
        for head in heads_of_layer:      # n_tokens × n_tokens
            activations_per_head = head.copy()
            # (i,j) = how much (activations coming ultimately from) token i influences token j
            if layer_norm:       # Normalize influence (across all tokens i) on each token j
                for j in range(0, len(activations_per_head)):
                    activations_per_head[:,j] = normalize(activations_per_head[:, j])
            summed_activations += activations_per_head

        mean_activations_per_layer.append(summed_activations/heads_per_layer.shape[1])

    return mean_activations_per_layer

bert_version = 'bert-base-cased'    # TODO Why no case?

tokenizer = BertTokenizer.from_pretrained(bert_version)

METHOD = "cat" # or "mat"
LAYER_NORM = True
    # What about normalizing per layer, instead of per head? Does that make any sense? Yes, a bit.
    # However, since BERT has LAYER NORM in each attention head, outputs of all heads will have same mean/variance.
    # Does this mean that all heads will contribute same amount of information? Yes, roughly.


# LAYERS = [[0,1,2], [3,4,5], [6,7,8], [9,10,11]]
# LAYERS = [[0,1,2,3,4,5,6,7,8,9,10,11]]
LAYERS = [0,1,2,3,4,5,6,7,8,9,10,11]
# LAYERS = [layer_inds[:i] for i in range(1,13)]
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

measures_per_layer = []

for sequence in DATA:

    tokens_a, tokens_b, attn = attention_visualizer.get_viz_data(sequence)
    all_tokens = tokens_a + tokens_b
    attn = attn.squeeze()

    measure_per_layer = (compute_PAT if METHOD == "pat" else compute_MAT)(attn, layer_norm=LAYER_NORM)
    measures_per_layer.append([pd.DataFrame(measure, index=all_tokens, columns=all_tokens) for measure in measure_per_layer])
    # TODO: possibility of cumulative MAT? No; if need be I can allow plotting averages over multiple layers.

# TODO Now sum over sequences of the same type.

for dfs, layer in zip(zip(measures_per_layer[0], measures_per_layer[1]), LAYERS):

    fig, axs = plt.subplots(ncols=len(dfs) + 1, figsize=(12, 4))
    plt.subplots_adjust(wspace = .6, top = .9)
    fig.suptitle("Layer {}".format(layer))

    # Individual sequence plots
    vmin = min([df.min().min() for df in dfs])
    vmax = max([df.max().max() for df in dfs])

    for i, df in enumerate(dfs):
        # (i,j) --> (j,i): how much j is unfluenced by i
        ax = sns.heatmap(df.transpose(), xticklabels=True, yticklabels=True, vmin=vmin, vmax=vmax, linewidth=0.5, ax=axs[i], cbar=False, cmap="Blues", square=True, cbar_kws={'shrink':.5}, label='small')
        ax.xaxis.tick_top()
        plt.setp(ax.get_xticklabels(), rotation=90)

    ## Difference plot
    diff = dfs[0] - dfs[1].values
    # TODO change values to be only shared tokens
    vmin, vmax = diff.min().min(), diff.max().max()
    vmin = -(max(0-vmin, vmax))
    vmax = (max(0-vmin, vmax))

    ax = sns.heatmap(diff.transpose(), xticklabels=True, yticklabels=True, vmin=vmin, vmax=vmax, cbar=False, linewidth=0.5, ax=axs[i+1], cmap="coolwarm_r", square=True, cbar_kws={'shrink':.5}, label='small')
    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), rotation=90)

    pylab.savefig("output/temp{}.png".format(layer))
    # pylab.show()


# import imageio
# images = []
# for filename in filenames:
#     images.append(imageio.imread(filename))
# imageio.mimsave('/path/to/movie.gif', images, format='GIF', duration=5)