from nltk.parse.viterbi import ViterbiParser
from nltk.corpus import LazyCorpusLoader, BracketParseCorpusReader
from nltk import PCFG
from nltk.grammar import ProbabilisticProduction
from nltk import Tree, Nonterminal
from nltk.treetransforms import chomsky_normal_form
import numpy as np


def simplify_functional_tag(tag):
    if '-' in tag:
        tag = tag.split('-')[0]
    return tag


def get_tag(tree):
    if isinstance(tree, Tree):
        return Nonterminal(simplify_functional_tag(tree.label()))
    else:
        return tree


def tree_to_production(tree):
    return ProbabilisticProduction(get_tag(tree), [get_tag(child) for child in tree], **{'prob': 0})


def tree_to_productions(tree):
    yield tree_to_production(tree)
    for child in tree:
        if isinstance(child, Tree):
            for prod in tree_to_productions(child):
                yield prod


treebank = LazyCorpusLoader('treebank/combined', BracketParseCorpusReader, r'wsj_.*\.mrg')


def get_productions(productions):

    probabilities = dict()
    productions_to_return = list(set(productions))

    for prod in productions:
        if str(prod) in probabilities:
            probabilities[str(prod)] += 1
        else:
            probabilities[str(prod)] = 1

    lhs_of_prods = set([prod.lhs() for prod in productions])

    for lhs in lhs_of_prods:
        number_of_occurrences = 0
        for prob in probabilities:
            if prob.startswith(str(lhs) + " "):
                number_of_occurrences += probabilities[prob]
        for prob in probabilities:
            if prob.startswith(str(lhs) + " "):
                probabilities[prob] = probabilities[prob] / number_of_occurrences

    for index in range(len(productions_to_return)):
        prod = productions_to_return[index]
        productions_to_return[index] = ProbabilisticProduction(prod.lhs(), prod.rhs(),
                                                               **{'prob': probabilities[str(prod)]})

    return productions_to_return


def pcfg_learn(treebank, n):
    productions = list()
    for i in range(n):
        for tree in treebank.parsed_sents()[:i+1]:
            chomsky_normal_form(tree, factor='right', horzMarkov=1, vertMarkov=1, childChar='|', parentChar='^')
            prod_gen = tree_to_productions(tree)
            tree_to_append = next(prod_gen)
            while tree_to_append:
                productions.append(tree_to_append)
                try:
                    tree_to_append = next(prod_gen)
                except Exception as e:
                    tree_to_append = False
    productions = get_productions(productions)
    return PCFG(Nonterminal('S'), productions)


pcfg_training = pcfg_learn(treebank, 400)
parser = ViterbiParser(pcfg_training)


def get_list_of_labelled_constituents(parse_tree, lst=None, first_index=None, last_index=None):
    if lst is None:
        return get_list_of_labelled_constituents(parse_tree, list(), 0, len(parse_tree.leaves()) - 1)
    if not len(lst):
        lst = [(parse_tree.label(), first_index, last_index)]
    if len(list(parse_tree.subtrees())) == 1:
        return [(parse_tree.label(), first_index, last_index)]
    else:
        for child in parse_tree:
            labelled_constituents = (child.label(), first_index, first_index + len(child.leaves()) - 1)
            get_list_of_labelled_constituents(child, lst, first_index, first_index + len(child.leaves()) - 1)
            first_index += len(child.leaves())
            last_index += len(child.leaves())
            lst.append(labelled_constituents)
        return lst


def get_test_dataset(treebank, n):
    test_trees = list()
    for i in range(400, n+400):
        for tree in treebank.parsed_sents()[:i+1]:
            chomsky_normal_form(tree, factor='right', horzMarkov=1, vertMarkov=1, childChar='|', parentChar='^')
            test_trees.append(tree)
        if len(test_trees) == 50:
            return test_trees
    return test_trees


def turn_tree_into_sentence(tree):
    sentence = ""
    if isinstance(tree, Tree):
        for child in tree:
            sentence += turn_tree_into_sentence(child)
    else:
        sentence += tree + ' '
    return sentence


trees = get_test_dataset(treebank, 50)
sentences = list()
for tree in trees:
    sentences.append(turn_tree_into_sentence(tree))
    if len(sentences) == 50:
        break
count = 0
gen_trees = list()
for sentence in sentences:
    gen_tree = parser.parse_all(sentence.split())
    if len(gen_tree):
        gen_trees.append(gen_tree[0])

list_of_labelled_constituents_for_trees = list()
for tree in trees:
    list_of_labelled_constituents_for_trees.append(get_list_of_labelled_constituents(tree))
    if len(list_of_labelled_constituents_for_trees) == 50:
        break

list_of_labelled_constituents_for_gen_trees = list()
for tree in gen_trees:
    list_of_labelled_constituents_for_gen_trees.append(get_list_of_labelled_constituents(tree))
    if len(list_of_labelled_constituents_for_gen_trees) == 50:
        break


def get_unlabelled_precision_and_recall(labelled_constituents1, labelled_constituents2):
    precisions = list()
    recalls = list()
    for index in range(len(labelled_constituents1)):
        labelled_constituent1 = labelled_constituents1[index]
        labelled_constituent2 = labelled_constituents2[index]
        number_of_correct_constituents = 0
        number_of_total_constituents = len(labelled_constituent1)
        number_of_total_constituents2 = len(labelled_constituent2)
        for index2 in range(len(labelled_constituent1)):
            try:
                if labelled_constituent1[index2][1] == labelled_constituent2[index2][1]:
                    if labelled_constituent1[index2][2] == labelled_constituent2[index2][1]:
                        number_of_correct_constituents += 1
            except Exception as e:
                continue
        precisions.append(number_of_correct_constituents / number_of_total_constituents)
        recalls.append(number_of_correct_constituents / number_of_total_constituents2)
    return [np.mean(precisions), np.mean(recalls),
            (2*np.mean(precisions)*np.mean(recalls)) / (np.mean(precisions) + np.mean(recalls))]


def get_labelled_precision_and_recall(labelled_constituents1, labelled_constituents2):
    precisions = list()
    recalls = list()
    for index in range(len(labelled_constituents1)):
        labelled_constituent1 = labelled_constituents1[index]
        labelled_constituent2 = labelled_constituents2[index]
        number_of_correct_constituents = 0
        number_of_total_constituents = len(labelled_constituent1)
        number_of_total_constituents2 = len(labelled_constituent2)
        for index2 in range(len(labelled_constituent1)):
            try:
                if labelled_constituent1[index2][0] == labelled_constituent2[index2][0]:
                    number_of_correct_constituents += 1
            except Exception as e:
                continue
        precisions.append(number_of_correct_constituents / number_of_total_constituents)
        recalls.append(number_of_correct_constituents / number_of_total_constituents2)
    return [np.mean(precisions), np.mean(recalls),
            (2*np.mean(precisions)*np.mean(recalls)) / (np.mean(precisions) + np.mean(recalls))]


print(get_labelled_precision_and_recall(list_of_labelled_constituents_for_gen_trees,
                                        list_of_labelled_constituents_for_trees))

print(get_unlabelled_precision_and_recall(list_of_labelled_constituents_for_gen_trees,
                                          list_of_labelled_constituents_for_trees))
