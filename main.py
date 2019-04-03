from bertviz import attention, visualization
from bertviz.pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pylab as pylab

import pandas as pd

import os

import csv

import itertools
import imageio
import warnings

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
        # for the next layer, use summed_activations as the next input activations
        percolated_activations = summed_activations
        # I believe normalizing the activations (as a whole or per col) makes no difference.

        percolated_activations_per_layer.append(percolated_activations)

    return percolated_activations_per_layer


def compute_MAT(heads_per_layer, layer_norm=True):
    """
    Computes Mean Attention per Token (MAT), i.e,, mean across all layers and heads.
    :param heads_per_layer:
    :param layer_norm:
    :return: Averages (across heads) per layer
    """
    mean_activations_per_layer = []
    summed_mean_activations = np.zeros_like(heads_per_layer[0][1])
    for heads_of_layer in heads_per_layer:
        summed_activations = np.zeros_like(heads_per_layer[0][1])
        for head in heads_of_layer:      # n_tokens × n_tokens
            activations_per_head = head.copy()
            # (i,j) = how much (activations coming from) token i influences token j
            if layer_norm:       # Normalize influence (across all tokens i) on each token j
                for j in range(0, len(activations_per_head)):
                    activations_per_head[:,j] = normalize(activations_per_head[:, j])
            summed_activations += activations_per_head

        mean_activations_per_layer.append(summed_activations/heads_per_layer.shape[1])

    return mean_activations_per_layer


bert_version = 'bert-base-cased'    # TODO Why no cased model available? Is this an older BERT version?

TRANSPOSE = True    # True to plot as "rows influenced by cols" (otherwise: rows influencing cols).
OUTPUT_GIF = False
METHOD = "PAT" # "PAT" or "MAT"     # TODO Allow cumulative MAT too... CAT?
GROUPED = True
LAYER_NORM = True
    # What about normalizing per layer, instead of per head? Does that make any sense? Yes, a bit.
    # However, since BERT has LAYER NORM in each attention head, outputs of all heads will have same mean/variance.
    # Does this mean that all heads will contribute same amount of information? Yes, roughly.

PLOT_DIFFERENCES = True
FACTORS_TO_PLOT = ['reflexivity', 'gender']
if len(FACTORS_TO_PLOT) > 2:
    print("WARNING: Cannot plot more than 2 factors at a time. Trimming to", FACTORS_TO_PLOT[:2])
    FACTORS_TO_PLOT = FACTORS_TO_PLOT[:2]


DATA = [
        # "The boy has a cat while the girl has a pigeon.",
        # "The boy has a cat while the girl has a pigeon."
        # "The boy has a cat. He likes to stroke it.",
        # "The boy has no cat. He likes to stroke it.",
        'reflexive, masculine, |0 The teacher | wants |1 every boy | to like |2 himself.',
        'irreflexive, masculine, |0 The teacher | wants |1 every boy | to like |2 him.',
        'reflexive, feminine, |0 The officers | want |1 all drivers | to like |2 themselves.',
        'irreflexive, feminine, |0 The officers | want |1 all drivers | to like |2 them.',
        # "I cannot find one of my ten marbles. It's probably under the couch.",
        # "I only found nine of my ten marbles. It's probably under the couch.",
        # '|0 Every farmer | who |1 owns a donkey |2 beats it.',
        # '|0 No farmer | who |1 owns a donkey |2 beats it.',
        # "Few of the children ate their ice-cream. They ate the apple flavor first.",
        # "Few of the children ate their ice-cream. They threw it around the room instead.",
        # "Few of the children ate their ice-cream. The others threw it around the room instead.",
    ]

def parse_data(data, tokenizer, factor_legend=None, group_legend=None):

    items = []
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

        items.append([s.strip() for s in row[:-1]] + [sentence.strip()] + [' '.join(['[CLS]'] + tokenizer.tokenize(sentence) + ['[SEP]'])] + token_ids_list)

    if group_legend is None:
        group_names = ['g{}'.format(i) for i in range(max_group_id + 1)]
    else:
        group_names = [group_legend[key] for key in group_legend]

    if factor_legend is None:
        factor_names = ['f{}'.format(i) for i in range(num_factors)]
    else:
        factor_names = [factor_legend[key] for key in factor_legend]

    columns = factor_names + ['sentence'] + ['tokenized'] + group_names

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        items = pd.DataFrame(items, columns=columns)
        items.num_factors = num_factors
        items.factors = factor_names
        items.num_groups = max_group_id + 1
        items.groups = group_names
        items.levels = {factor: items[factor].unique().tolist() for factor in items.factors}
        items.conditions = list(itertools.product(*[items.levels[factor] for factor in items.factors]))

    return items


tokenizer = BertTokenizer.from_pretrained(bert_version)

items = parse_data(DATA, tokenizer, {0: 'reflexivity', 1: 'gender'}, {0: 'subject', 1: 'object', 2: 'anaphor'})

model = BertModel.from_pretrained(bert_version)
n_layers = len(model.encoder.layer)
# TODO Bypass the AttentionVisualizer code altogether and remove from repo; I'm not really using it.
attention_visualizer = visualization.AttentionVisualizer(model, tokenizer)

## Compute attention weights, one item at a time
weights_for_all_items = []
for _, each_item in items.iterrows():

    tokens_a, tokens_b, attention = attention_visualizer.get_viz_data(each_item['sentence'])
    all_tokens = tokens_a + tokens_b
    attention = attention.squeeze()

    weights_per_layer = (compute_PAT if METHOD == "PAT" else compute_MAT)(attention, layer_norm=LAYER_NORM)

    # TODO Put the following outside the current loop, inside its own (anticipating intermediary writing of results to disk)
    # Take averages over groups of tokens
    if GROUPED:
        grouped_weights_per_layer = []
        for m in weights_per_layer:
            # Group horizontally
            grouped_weights_horiz = []
            for group in items.groups:
                # TODO do I need to check if not None, in case not all items have all groups?
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
df = pd.concat([items, weights_for_all_items], axis=1)


## Compute means for all conditions
means = df.groupby(items.factors).mean().values
means = means.reshape(len(items.conditions) * n_layers, -1)
multi_index = pd.MultiIndex.from_product([items.levels[factor] for factor in items.factors] + [list(range(n_layers))], names=items.factors + ['layer'])
df_means = pd.DataFrame(means, index=multi_index)


## Prepare creating outputs
# TODO More meaningful output names
out_path = 'output/temp'
out_path_idx = 0
while os.path.exists(out_path):
    out_path_idx += 1
    out_path = 'output/temp_{}'.format(out_path_idx)
os.mkdir(out_path)
out_filepaths = [] # for keeping track to create gif

# Global min/max to have same color map everywhere
vmin = df_means.min().min()
vmax = df_means.max().max()


# Means for selected conditions
factors_to_plot = FACTORS_TO_PLOT
levels_horiz = items.levels[factors_to_plot[0]]
levels_vert = items.levels[factors_to_plot[1]] if len(factors_to_plot) == 2 else [None]

if len(levels_horiz) == 2 and PLOT_DIFFERENCES:  # if two levels, also compute difference
    levels_horiz.append('<DIFF>')
if len(levels_vert) == 2 and PLOT_DIFFERENCES:  # if two levels, also compute difference
    levels_vert.append('<DIFF>')

n_plots_horiz = len(levels_horiz)
n_plots_vert = len(levels_vert)

df_means = df_means.groupby(factors_to_plot + ['layer']).mean()

# TODO allow plotting means over layers
for l in range(n_layers):

    fig, axs = plt.subplots(ncols=n_plots_horiz, nrows=n_plots_vert, figsize=(4 * n_plots_horiz, 4 * n_plots_vert))
    plt.subplots_adjust(wspace=.6, top=.9)
    fig.suptitle("{}-scores given {} (layer {})".format(METHOD, ' × '.join(factors_to_plot), l))

    # Keep for computing differences
    weights_for_diff = [[None,None],
                       [None,None]]

    for h, level_horiz in enumerate(levels_horiz):

        for v, level_vert in enumerate(levels_vert):

            index = items.groups if GROUPED else None # TODO Replace None by exemplary tokens... items.iloc[0].tokenized.split()

            is_difference_plot = True
            if level_horiz != "<DIFF>" and level_vert == "<DIFF>":
                weights = weights_for_diff[h][0] - weights_for_diff[h][1]
            elif level_horiz == "<DIFF>" and level_vert != "<DIFF>":
                weights = weights_for_diff[0][v] - weights_for_diff[1][v]
            elif level_horiz == "<DIFF>" and level_vert == "<DIFF>":
                weights = weights_for_diff[0][0] - weights_for_diff[1][1]
            else:
                is_difference_plot = False
                weights = df_means.loc[(level_horiz, level_vert, l)] if level_vert is not None else df_means.loc[(level_horiz, l)]
                weights = weights.values
                dim = int(np.sqrt(weights.shape[-1]))
                weights = weights.reshape(dim, dim)
                if TRANSPOSE:
                    weights = weights.transpose()
                weights = pd.DataFrame(weights, index=index, columns=index)
                # Keep for difference plot
                weights_for_diff[h][v] = weights

            # TODO Consider setting global vmin/vmax only in case of MAT; in that case also for is_difference_plot.
            ax = sns.heatmap(weights,
                             xticklabels=True,
                             yticklabels=True,
                             vmin=vmin if not is_difference_plot else None,
                             vmax=vmax if not is_difference_plot else None,
                             center=0 if is_difference_plot else None,
                             linewidth=0.5,
                             ax=axs[h,v] if level_vert is not None else axs[h],
                             cbar=False,
                             cmap="coolwarm_r" if is_difference_plot else "Blues",
                             square=True,
                             cbar_kws={'shrink':.5},
                             label='small')
            if is_difference_plot:
                ax.set_title('Difference')
            else:
                ax.set_title('{} & {}'.format(level_horiz, level_vert) if level_vert is not None else level_horiz)
            # ax.xaxis.tick_top()
            plt.setp(ax.get_yticklabels(), rotation=0)

    out_filepath = "{}/{}_{}_layer{}.png".format(out_path, METHOD, '-x-'.join(factors_to_plot), l)
    print("Saving figure:", out_filepath)
    pylab.savefig(out_filepath)
    # pylab.show()

    out_filepaths.append(out_filepath)


if OUTPUT_GIF:  # :)
    out_filepath = "{}/{}_{}_animated.gif".format(out_path, METHOD, '-x-'.join(factors_to_plot))
    images = []
    for filename in out_filepaths:
        images.append(imageio.imread(filename))
    imageio.mimsave(out_filepath, images, format='GIF', duration=.5)
    print("Saving movie:", out_filepath)