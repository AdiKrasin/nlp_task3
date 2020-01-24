from nltk.corpus import LazyCorpusLoader, BracketParseCorpusReader
from nltk import PCFG
from nltk.grammar import ProbabilisticProduction
from nltk import Tree, Nonterminal
from nltk.probability import FreqDist, MLEProbDist
import math


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


def tree_to_productions(tree, father_title):
    yield tree_to_production(tree), father_title
    for child in tree:
        if isinstance(child, Tree):
            for prod in tree_to_productions(child, tree.label()):
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
    dist = FreqDist(productions_to_return)
    #dist.plot(len(probabilities))

    return productions_to_return, dist


def pcfg_learn1(treebank, n):
    productions = list()
    for i in range(n):
        for tree in treebank.parsed_sents()[:i+1]:
            prod_gen = tree_to_productions(tree, "BOT")
            tree_to_append = next(prod_gen)[0]
            while tree_to_append:
                if tree_to_append.lhs() == Nonterminal('NP'):
                    productions.append(tree_to_append)
                try:
                    tree_to_append = next(prod_gen)[0]
                except Exception as e:
                    tree_to_append = False
    productions, dist = get_productions(productions)
    return PCFG(Nonterminal('NP'), productions), dist


def pcfg_learn2(treebank, n):
    productions = list()
    for i in range(n):
        for tree in treebank.parsed_sents()[:i+1]:
            prod_gen = tree_to_productions(tree, "BOT")
            tree_to_append, father_title = next(prod_gen)
            while tree_to_append:
                if tree_to_append.lhs() == Nonterminal('NP') and father_title == 'S':
                    productions.append(tree_to_append)
                try:
                    tree_to_append, father_title = next(prod_gen)
                except Exception as e:
                    tree_to_append = False
    productions, dist = get_productions(productions)
    return PCFG(Nonterminal('NP'), productions), dist


def pcfg_learn3(treebank, n):
    productions = list()
    for i in range(n):
        for tree in treebank.parsed_sents()[:i+1]:
            prod_gen = tree_to_productions(tree, "BOT")
            tree_to_append, father_title = next(prod_gen)
            while tree_to_append:
                if tree_to_append.lhs() == Nonterminal('NP') and father_title == 'VP':
                    productions.append(tree_to_append)
                try:
                    tree_to_append, father_title = next(prod_gen)
                except Exception as e:
                    tree_to_append = False
    productions, dist = get_productions(productions)
    return PCFG(Nonterminal('NP'), productions), dist


dist1 = pcfg_learn1(treebank, 200)[1]
dist2 = pcfg_learn2(treebank, 200)[1]
dist3 = pcfg_learn3(treebank, 200)[1]

mle_1 = MLEProbDist(dist1)
mle_2 = MLEProbDist(dist2)
mle_3 = MLEProbDist(dist3)


def compute_kl_divergence(mle_dist1, mle_dist2):
    ans = 0
    for p in mle_dist1.freqdist():
        for q in mle_dist2.freqdist():
            if p.rhs() == q.rhs():
                ans += p.prob() * math.log(p.prob() / q.prob())
    return ans


print(compute_kl_divergence(mle_1, mle_2))
print(compute_kl_divergence(mle_1, mle_3))
print(compute_kl_divergence(mle_2, mle_3))
