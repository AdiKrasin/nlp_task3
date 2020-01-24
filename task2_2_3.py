from nltk.corpus import LazyCorpusLoader, BracketParseCorpusReader
from nltk import PCFG
from nltk.grammar import ProbabilisticProduction
from nltk import Tree, Nonterminal
from nltk.treetransforms import chomsky_normal_form


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

'''
# Raw form
print(treebank.parsed_sents()[:1])

# Pretty print
print(treebank.parsed_sents()[0])

# we need to transform the tree to remove NONE tags and simplify tags.
'''


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
            # this is just for 2.3
            # chomsky_normal_form(tree, factor='right', horzMarkov=1, vertMarkov=1, childChar='|', parentChar='^')
            productions.append(next(tree_to_productions(tree)))
    productions = get_productions(productions)
    return PCFG(Nonterminal('S'), productions)


print(pcfg_learn(treebank, 200))
