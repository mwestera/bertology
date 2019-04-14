from collections import defaultdict, namedtuple

from numpy import argmax

Arc = namedtuple('Arc', ('head', 'weight', 'tail'))


def max_sa_from_nodes(arcs, nodes):

    trees = []
    values = []

    for source in nodes:
        tree = max_spanning_arborescence(arcs, source)
        if not len(tree) == 0:
            trees.append(tree)
            values.append(sum([arc[1] for arc in tree.values()]))

    maxidx = argmax(values)

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
            if value == value:      # fails if NaN
                arcs.append(Arc(i,value,j))
    return arcs


def arcs_to_tuples(arcs):
    tuples = []
    sum = 0
    for arc in arcs:
        tuples.append((arc[0],arc[2]))
        sum += arc[1]
    return tuples, sum


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

    if nodes1 != nodes2:
        print("Something's wrong.")

    score = len(set(tree1).intersection(set(tree2))) / (len(nodes1) - 1.0)

    return score


# The recall metric is ignored when evaluating syntactic trees because all tokens are being labeled in one way or another. There are 5 most common metrics for the evaluation of syntactic dependency parsing:
#
#     (Unlabeled)Head Attachment Score (percent of nodes which are correctly attached to their parent)
#     Label Precision (percent of nodes whose dependency labeled is predicted correctly)
#     Labeled Attachment Score (percent of node for which both of the above are true)
#     Branch Precision (percent of the Paths (from root to leaf) that are being classified correctly)
#     Correct trees precision (percent of the sentences from the eval corpus which have been parsed flawlessly)
#
