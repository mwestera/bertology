from bertviz import attention, visualization
from bertviz.pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pylab as pylab

import pandas as pd

import csv

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
        'reflexive, |0 The teacher | wants |1 every boy | to like |2 himself.',
        'plain, |0 The teacher | wants |1 every boy | to like |2 him.',
        # "I cannot find one of my ten marbles. It's probably under the couch.",
        # "I only found nine of my ten marbles. It's probably under the couch.",
        # '|0 Every farmer | who |1 owns a donkey |2 beats it.',
        # '|0 No farmer | who |1 owns a donkey |2 beats it.',
        # "Few of the children ate their ice-cream. They ate the apple flavor first.",
        # "Few of the children ate their ice-cream. They threw it around the room instead.",
        # "Few of the children ate their ice-cream. The others threw it around the room instead.",
    ]

def parse_data(data):

    parsed_data = []
    num_factors = None
    max_group_id = 0

    for row in csv.reader(data):
        num_factors = len(row)-1    # Num rows minus the sentence itself
        group_to_token_ids = {}
        sentence = ""
        total_len = 1   # Take mandatory CLS symbol into account
        for each_part in row[-1].strip('|').split('|'):
            first_char = each_part[0]
            if first_char.isdigit():
                group_id = int(first_char)
                each_part = each_part[1:].strip()
                max_group_id = max(max_group_id, group_id)
            tokens = tokenizer.tokenize(each_part)
            if first_char.isdigit():
                if group_id in group_to_token_ids:
                    group_to_token_ids[group_id].append(list(range(total_len, total_len + len(tokens))))
                else:
                    group_to_token_ids[group_id] = list(range(total_len, total_len + len(tokens)))
            total_len += len(tokens)
            sentence += each_part.strip() + ' '


        # collect token group ids in a list instead of dict
        token_ids_list = [[] for _ in range(max(group_to_token_ids)+1)]
        for key in group_to_token_ids:
            token_ids_list[key] = group_to_token_ids[key]

        parsed_data.append(row[:-1] + [sentence.strip()] + [' '.join(['CLS'] + tokenizer.tokenize(sentence) + ['SEP'])] + token_ids_list)

    columns = ['f{}'.format(i) for i in range(num_factors)] + ['sentence'] + ['tokenized'] + ['g{}'.format(i) for i in range(max_group_id + 1)]

    parsed_data = pd.DataFrame(parsed_data, columns=columns)
    parsed_data.num_factors = num_factors
    parsed_data.factors = ['f{}'.format(i) for i in range(num_factors)]
    parsed_data.num_groups = max_group_id + 1
    parsed_data.groups = ['g{}'.format(i) for i in range(max_group_id + 1)]

    return parsed_data


data = parse_data(DATA)
print(data)

# IDEA: Compare quantifiers, restrictor vs scope, to see how the info flows to the quantifier... advantage: very uniform sentences...

model = BertModel.from_pretrained(bert_version)
attention_visualizer = visualization.AttentionVisualizer(model, tokenizer)

# First, I want to compare pairs of sentences across layers. Only afterwards think about accumulating stats of multiple sentences.

measures_per_layer = []

for i, row in data.iterrows():

    tokens_a, tokens_b, attn = attention_visualizer.get_viz_data(row['sentence'])
    all_tokens = tokens_a + tokens_b
    attn = attn.squeeze()

    measure_per_layer = (compute_PAT if METHOD == "pat" else compute_MAT)(attn, layer_norm=LAYER_NORM)

    grouped_measure_per_layer = []

    for m in measure_per_layer:

        grouped_measure = []

        for group in data.groups:
            # TODO check if not None?
            grouped_measure.append(m[row[group]].mean(axis=0))

        grouped_measure = np.stack(grouped_measure)
        grouped_measure2 = []

        for group in data.groups:
            grouped_measure2.append(grouped_measure[:,row[group]].mean(axis=1))

        grouped_measure = np.stack(grouped_measure2).transpose()

        grouped_measure_per_layer.append(pd.DataFrame(grouped_measure, index=data.groups, columns=data.groups))

    measures_per_layer.append(grouped_measure_per_layer)

    # TODO: possibility of cumulative MAT? No; if need be I can allow plotting averages over multiple layers.

# measures_per_layer now contains num_seq * dataframe [num_groups * num_groups]


# TODO Now sum over sequences of the same type.   data.factors



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