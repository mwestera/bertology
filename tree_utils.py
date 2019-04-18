from collections import defaultdict, namedtuple

import numpy as np

import data_utils

from scipy.stats.stats import pearsonr

Arc = namedtuple('Arc', ('head', 'weight', 'tail'))


def max_sa_from_nodes(arcs, nodes):

    trees = []
    values = []

    for source in nodes:
        tree = max_spanning_arborescence(arcs, source)
        if not len(tree) == 0:
            trees.append(tree)
            values.append(sum([arc[1] for arc in tree.values()]))

    maxidx = np.argmax(values)

    return trees[maxidx], values[maxidx]


def max_spanning_arborescence(arcs, source):
    arcs = [Arc(arc[0], -arc[1], arc[2]) for arc in arcs]
    result = min_spanning_arborescence(arcs, source)
    return {key:Arc(arc[0], -arc[1], arc[2]) for key,arc in result.items()}

def min_spanning_arborescence(arcs, source):
    # This and functions called herein from David Eisenstat's answer at https://stackoverflow.com/a/34407749/11056813

    good_arcs = []
    quotient_map = {arc.tail: arc.tail for arc in arcs}
    quotient_map[source] = source
    while True:
        min_arc_by_tail_rep = {}
        successor_rep = {}
        for arc in arcs:
            if arc.tail == source:
                continue
            tail_rep = quotient_map[arc.tail]
            head_rep = quotient_map[arc.head]
            if tail_rep == head_rep:
                continue
            if tail_rep not in min_arc_by_tail_rep or min_arc_by_tail_rep[tail_rep].weight > arc.weight:
                min_arc_by_tail_rep[tail_rep] = arc
                successor_rep[tail_rep] = head_rep
        cycle_reps = find_cycle(successor_rep, source)
        if cycle_reps is None:
            good_arcs.extend(min_arc_by_tail_rep.values())
            return spanning_arborescence(good_arcs, source)
        good_arcs.extend(min_arc_by_tail_rep[cycle_rep] for cycle_rep in cycle_reps)
        cycle_rep_set = set(cycle_reps)
        cycle_rep = cycle_rep_set.pop()
        quotient_map = {node: cycle_rep if node_rep in cycle_rep_set else node_rep for node, node_rep in quotient_map.items()}


def find_cycle(successor, source):
    visited = {source}
    for node in successor:
        cycle = []
        while node not in visited:
            visited.add(node)
            cycle.append(node)
            node = successor[node]
        if node in cycle:
            return cycle[cycle.index(node):]
    return None


def spanning_arborescence(arcs, source):
    arcs_by_head = defaultdict(list)
    for arc in arcs:
        if arc.tail == source:
            continue
        arcs_by_head[arc.head].append(arc)
    solution_arc_by_tail = {}
    stack = arcs_by_head[source]
    while stack:
        arc = stack.pop()
        if arc.tail in solution_arc_by_tail:
            continue
        solution_arc_by_tail[arc.tail] = arc
        stack.extend(arcs_by_head[arc.tail])
    return solution_arc_by_tail


def matrix_to_arcs(matrix):
    arcs = []
    for i,row in enumerate(matrix):
        for j,value in enumerate(row):
            if i != j and value == value:      # fails if NaN
                arcs.append(Arc(i,value,j))
    return arcs


def arcs_to_tuples(arcs):
    tuples = []
    sum = 0
    for arc in arcs:
        tuples.append((arc[0],arc[2]))
        sum += arc[1]
    return tuples, sum


def conllu_to_arcs(tree):

    nodes_to_explore = [tree]
    arcs = []
    while len(nodes_to_explore) > 0:
        node = nodes_to_explore.pop()
        for c in node.children:
            nodes_to_explore.append(c)
            arcs.append((node.token['id'] - 1, c.token['id'] - 1))  # make zero-based

    # nodes = list(set([a[j] for j in [0, 1] for a in arcs]))
    # nodes.sort()

    return arcs


def tree_value_from_matrix(arcs, matrix):
    sum = 0
    for arc in arcs:
        sum += matrix[arc[0],arc[-1]]
    return sum


# arcs = matrix_to_arcs([[1,2,3,4,5],[6,5,4,3,2],[9,8,7,6,5],[1,2,3,6,4],[2,4,7,4,2]])
# print(arcs_to_tuples(min_spanning_arborescence(arcs, 0).values()))

def head_attachment_score(tree1, tree2):
    nodes1 = set([a[i] for i in [0,1] for a in tree1])
    nodes2 = set([a[i] for i in [0,1] for a in tree2])

    if len(tree2) == 0:
        return np.nan

    # This -1 works only if it is indeed a tree. But what if it's tree fragments...
    score = len(set(tree1).intersection(set(tree2))) / (len(tree2))  # Take nodes from tree 2 as true

    return score


def undirected_attachment_score(tree1, tree2):
    tree1 = [(a, b) if a < b else (b,a) for (a,b) in tree1]
    tree2 = [(a, b) if a < b else (b,a) for (a,b) in tree2]

    return head_attachment_score(tree1, tree2)


def pearson_correlation(matrix1, matrix2):

    # TODO Not sure if necessary:
    matrix1 = matrix1.reshape(-1)
    matrix2 = matrix2.reshape(-1)

    # Filter out nans
    matrix1_nonan = matrix1[~(np.isnan(matrix1) or np.isnan(matrix2))]
    matrix2_nonan = matrix1[~(np.isnan(matrix1) or np.isnan(matrix2))]

    return pearsonr(matrix1_nonan, matrix2_nonan)


def pearson_scores(matrix1, conllu_rep, matrix2=None, matrix2_bidirectional=None):
    if matrix2 is None:
        matrix2 = - arcs_to_distance_matrix(conllu_to_arcs(conllu_rep.to_tree()), False)
    if matrix2_bidirectional is None:
        matrix2_bidirectional = - arcs_to_distance_matrix(conllu_to_arcs(conllu_rep.to_tree()), False)

    # matrix1_irreflexive = matrix1.copy()
    # np.fill_diagonal(matrix1_irreflexive, np.nan)

    c1, p1 = pearson_correlation(matrix1, matrix2)
    c2, p2 = pearson_correlation(matrix1, matrix2_bidirectional)

    return [c1,p1,c2,p2]

    # pearson_correlation(matrix1_irreflexive, matrix2)




def filtered_scores(tree1, conllu_rep):

    categories = {

        # some based on token["upostag"]
        "upostag" : { 'open': data_utils.CONLLU_tags.open_class_tags,
                        'closed': data_utils.CONLLU_tags.closed_class_tags,
                      },

        # and some based on token["deprel"]
        "deprel": {'core': data_utils.CONLLU_tags.nominal_core_arguments,
                    'non-core': data_utils.CONLLU_tags.nominal_non_core_dependents + data_utils.CONLLU_tags.nominal_nominal_dependents,

                    # 'nominal': data_utils.CONLLU_tags.nominal_dependents,
                    # 'non-nominal': data_utils.CONLLU_tags.modifier_dependents + data_utils.CONLLU_tags.clause_dependents,
                   },
        }

    tree2 = conllu_to_arcs(conllu_rep.to_tree())

    scores = {}
    for feature in categories:
        for tags in categories[feature]:

            token_ids = []

            for token in conllu_rep:
                if token[feature] in categories[feature][tags]:
                    token_ids.append(token["id"]-1) # zero-based

            if feature == "upostag":
                tree1_filtered = [arc for arc in tree1 if (arc[0] in token_ids and arc[1] in token_ids)] # only arcs connecting such tags
                tree1_filtered_undirected = tree1_filtered  # In this case we don't want to look in both directions
                tree2_filtered = [arc for arc in tree2 if (arc[0] in token_ids and arc[1] in token_ids)]
            elif feature == "deprel":
                tree1_filtered = [arc for arc in tree1 if arc[1] in token_ids]
                tree1_filtered_undirected = [arc for arc in tree1 if (arc[1] in token_ids or arc[0] in token_ids)]  # look in both directions for relevant relations
                tree2_filtered = [arc for arc in tree2 if arc[1] in token_ids]

            scores[tags] = {
                'head_attachment_score': head_attachment_score(tree1_filtered, tree2_filtered),
                'undirected_attachment_score': undirected_attachment_score(tree1_filtered_undirected, tree2_filtered),
                'num_rels': len(tree2_filtered),
            }

    return scores


def arcs_to_distance_matrix(arcs, bidirectional=False):
    nodes = list(set([a[i] for i in [0, 1] for a in arcs]))

    if bidirectional:
        arcs = [(arc[i], arc[j]) for (i,j) in [(0,1),(1,0)] for arc in arcs]

    distances = np.zeros((len(nodes), len(nodes)), dtype=np.float)

    graph = arcs_to_graph(arcs)

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            path = find_shortest_path(graph, nodes[i], nodes[j])
            distances[i][j] = np.nan if path is None else len(path) - 1

    return distances


def arcs_to_graph(arcs):
    graph = {arc[0]: [] for arc in arcs}
    for arc in arcs:
        graph[arc[0]].append(arc[1])
    return graph


def find_shortest_path(graph, start, end, path=[]):
    """
    __source__='https://www.python.org/doc/essays/graphs/'
    __author__='Guido van Rossum'
    """
    path = path + [start]
    if start == end:
        return path
    if start not in graph:
        return None
    shortest = None
    for node in graph[start]:
        if node not in path:
            newpath = find_shortest_path(graph, node, end, path)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath
    return shortest


def get_scores(tree1, tree2):
    """
    :param tree1: list of pairs
    :param tree2: conllu-type tree
    :return: scores dictionary
    """
    tree2_arcs = conllu_to_arcs(tree2.to_tree())

    scores = filtered_scores(tree1, tree2)

    scores['all'] = {
        'head_attachment_score': head_attachment_score(tree1, tree2_arcs),
        'undirected_attachment_score': undirected_attachment_score(tree1, tree2_arcs),
        'num_rels': len(tree2_arcs)
    }

    return scores


# The recall metric is ignored when evaluating syntactic trees because all tokens are being labeled in one way or another. There are 5 most common metrics for the evaluation of syntactic dependency parsing:
#
#     (Unlabeled)Head Attachment Score (percent of nodes which are correctly attached to their parent)
#     Label Precision (percent of nodes whose dependency labeled is predicted correctly)
#     Labeled Attachment Score (percent of node for which both of the above are true)
#     Branch Precision (percent of the Paths (from root to leaf) that are being classified correctly)
#     Correct trees precision (percent of the sentences from the eval corpus which have been parsed flawlessly)
#
