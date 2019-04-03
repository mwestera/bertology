from bertviz import attention, visualization
from bertviz.pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pylab as pylab

import pandas as pd

import os

import csv

import imageio

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

METHOD = "pat" # "pat" or "mat"     # TODO Allow cumulative MAT too
GROUPED = True
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
        'reflexive, |0 The officers | want |1 all drivers | to like |2 themselves.',
        'plain, |0 The officers | want |1 all drivers | to like |2 them.',
        # "I cannot find one of my ten marbles. It's probably under the couch.",
        # "I only found nine of my ten marbles. It's probably under the couch.",
        # '|0 Every farmer | who |1 owns a donkey |2 beats it.',
        # '|0 No farmer | who |1 owns a donkey |2 beats it.',
        # "Few of the children ate their ice-cream. They ate the apple flavor first.",
        # "Few of the children ate their ice-cream. They threw it around the room instead.",
        # "Few of the children ate their ice-cream. The others threw it around the room instead.",
    ]

def parse_data(data, factor_legend=None, group_legend=None):

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

        parsed_data.append(row[:-1] + [sentence.strip()] + [' '.join(['[CLS]'] + tokenizer.tokenize(sentence) + ['[SEP]'])] + token_ids_list)

    if group_legend is None:
        group_names = ['g{}'.format(i) for i in range(max_group_id + 1)]
    else:
        group_names = [group_legend[key] for key in group_legend]

    if factor_legend is None:
        factor_names = ['f{}'.format(i) for i in range(num_factors)]
    else:
        factor_names = [factor_legend[key] for key in factor_legend]

    columns = factor_names + ['sentence'] + ['tokenized'] + group_names

    parsed_data = pd.DataFrame(parsed_data, columns=columns)
    parsed_data.num_factors = num_factors
    parsed_data.factors = factor_names
    parsed_data.num_groups = max_group_id + 1
    parsed_data.groups = group_names

    return parsed_data


items = parse_data(DATA, {0: 'anaphor_type'}, {0: 'subject', 1: 'object', 2: 'anaphor'})
print(items)

# IDEA: Compare quantifiers, restrictor vs scope, to see how the info flows to the quantifier... advantage: very uniform sentences...

model = BertModel.from_pretrained(bert_version)
attention_visualizer = visualization.AttentionVisualizer(model, tokenizer)

# TODO Compute these magic numbers
n_conditions = 2    # levels of all factors multiplied
n_layers = 12


# TODO Make this a global param; for plotting allow at most two factors? Compute diff only for 2 levels?
factors_to_plot = ['anaphor_type']


## Compute attention weights, one item at a time
weights_for_all_items = []
for _, each_item in items.iterrows():

    tokens_a, tokens_b, attention = attention_visualizer.get_viz_data(each_item['sentence'])
    all_tokens = tokens_a + tokens_b
    if ' '.join(all_tokens) != each_item['tokenized']:
        print('Warning!')
        print(' '.join(all_tokens))
        print(each_item['tokenized'])

    attention = attention.squeeze()

    weights_per_layer = (compute_PAT if METHOD == "pat" else compute_MAT)(attention, layer_norm=LAYER_NORM)

    # Take averages over groups of tokens
    if GROUPED:
        grouped_weights_per_layer = []
        for m in weights_per_layer:
            # Group horizontally
            grouped_weights_horiz = []
            for group in items.groups:
                # TODO check if not None?
                grouped_weights_horiz.append(m[each_item[group]].mean(axis=0))
            grouped_weights_horiz = np.stack(grouped_weights_horiz)

            # Group the result also vertically
            grouped_weights = []
            for group in items.groups:
                grouped_weights.append(grouped_weights_horiz[:, each_item[group]].mean(axis=1))
            grouped_weights = np.stack(grouped_weights).transpose() # transpose to restore original order

            # store
            grouped_weights_per_layer.append(grouped_weights)

        # stack and flatten for much easier handling with pandas, computing means etc.
        weights_per_layer = np.stack(grouped_weights_per_layer).reshape(-1)

    weights_for_all_items.append(weights_per_layer)

weights_for_all_items = pd.DataFrame(weights_for_all_items)

## Compute means for all conditions

# 1. Concatenate with items just so I can group by the different factors/levels, to compute means
df = pd.concat([items, weights_for_all_items], axis=1)
means = df.groupby(items.factors).mean().values

# 2. Now put them back into meaningful shape, with adequate multi-index
means = means.reshape(n_conditions * n_layers, -1)
multiindex = pd.MultiIndex.from_product([items[factor].unique() for factor in items.factors] + [list(range(n_layers))], names=items.factors + ['layer'])
df_means = pd.DataFrame(means, index=multiindex)

## Prepare creating outputs
# TODO More meaningful output names
out_path = 'output/temp'
out_path_idx = 0
while os.path.exists(out_path):
    out_path_idx += 1
    out_path = 'output/temp_{}'.format(out_path_idx)
os.mkdir(out_path)
out_filenames = [] # for keeping track to create gif

# Global min/max to have same color map everywhere
vmin = df_means.min().min()
vmax = df_means.max().max()

# TODO allow taking means over layers
for l in range(n_layers):

    # TODO Remove MAGIC everywhere below
    reflexive = pd.DataFrame(df_means.loc[('reflexive',l)].values.reshape(3,3), index=items.groups, columns=items.groups)
    plain = pd.DataFrame(df_means.loc[('plain',l)].values.reshape(3,3), index=items.groups, columns=items.groups)
    # TODO index should be either data.groups, or the tokens, depending on GROUPED.
    dfs_to_plot = [reflexive, plain]

    fig, axs = plt.subplots(ncols=2+1, figsize=(12, 4))
    plt.subplots_adjust(wspace = .6, top = .9)
    fig.suptitle("Layer {}".format(l))

    for each_item_index, df in enumerate(dfs_to_plot):
        # (i,j) --> (j,i): how much j is unfluenced by i
        ax = sns.heatmap(df.transpose(), xticklabels=True, yticklabels=True, vmin=vmin, vmax=vmax, linewidth=0.5, ax=axs[each_item_index], cbar=False, cmap="Blues", square=True, cbar_kws={'shrink':.5}, label='small')
        ax.xaxis.tick_top()
        plt.setp(ax.get_xticklabels(), rotation=90)

    ## Difference plot
    diff = dfs_to_plot[0] - dfs_to_plot[1].values
    # TODO change values to be only shared tokens
    # TODO Compute this globally, too
    vmin2, vmax2 = diff.min().min(), diff.max().max()
    vmin2 = -(max(0-vmin2, vmax2))
    vmax2 = (max(0-vmin2, vmax2))

    ax = sns.heatmap(diff.transpose(), xticklabels=True, yticklabels=True, center=0, vmin=vmin2, cbar=False, linewidth=0.5, ax=axs[each_item_index + 1], cmap="coolwarm_r", square=True, cbar_kws={'shrink':.5}, label='small')
    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), rotation=90)

    # TODO More meaningful output names
    out_filename = "{}/temp{}.png".format(out_path,l)
    print("Saving figure:", out_filename)
    pylab.savefig(out_filename)
    # pylab.show()

    out_filenames.append(out_filename)

# TODO more meaningful output name; tweak timing
images = []
for filename in out_filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('{}/temp.gif'.format(out_path), images, format='GIF', duration=.5)
print("Saving movie:", '{}/temp.gif'.format(out_path))