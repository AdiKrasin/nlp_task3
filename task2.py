from nltk.grammar import Nonterminal
from nltk.grammar import toy_pcfg2
from nltk.probability import DictionaryProbDist
from nltk.tree import Tree
import os
from nltk.grammar import induce_pcfg, ProbabilisticProduction, PCFG

productions_corpus = list()


def pcfg_generate(grammar):

    def non_terminal_into_terminal(non_terminal):
        nt_productions = grammar.productions(Nonterminal(str(non_terminal)))
        my_dict = dict()
        for pr in nt_productions: my_dict[pr.rhs()] = pr.prob()
        nt_productions_probDist = DictionaryProbDist(my_dict)
        genereted = nt_productions_probDist.generate()
        return list(genereted)

    def nts_into_ts(genereted_nts):
        for index in range(len(genereted_nts)):
            old_nt = genereted_nts[index]
            try:
                t = non_terminal_into_terminal(genereted_nts[index])
            except Exception as e:
                continue
            productions_corpus.append(ProbabilisticProduction(Nonterminal(old_nt), tuple(t), **{'prob': 0}))
            genereted_nts[index] = nts_into_ts(Tree(old_nt, t))
        return genereted_nts

    productions = grammar.productions()
    dic = dict()
    for pr in productions: dic[pr.rhs()] = pr.prob()
    productions_probDist = DictionaryProbDist(dic)
    genereted = productions_probDist.generate()
    productions_corpus.append(ProbabilisticProduction(Nonterminal('S'), genereted, **{'prob': 0}))
    genereted = Tree('S', [genereted[0], genereted[1]])
    return nts_into_ts(genereted)


corpus = list()
file_content = ""
for i in range(1000):
    res_tree = pcfg_generate(toy_pcfg2)
    file_content += str(res_tree) + "\n"
    corpus.append(res_tree)

if os.path.exists(".\\toy_pcfg2.gen"):
    os.remove(".\\toy_pcfg2.gen")
else:
    with open(".\\toy_pcfg2.gen", "w+") as f:
        f.write(file_content)


# this is the probability distribution for toy_pcfg2.
productions_toy_pcfg2 = toy_pcfg2.productions()
fd_toy_pcfg2 = induce_pcfg(Nonterminal('S'), productions_toy_pcfg2)

original_production_corpus = productions_corpus
productions_corpus = list(set(productions_corpus))
probabilities = dict()
for prod in original_production_corpus:
    if str(prod) in probabilities:
        probabilities[str(prod)] += 1
    else:
        probabilities[str(prod)] = 1

lhs_of_prods = set([prod.lhs() for prod in original_production_corpus])

for lhs in lhs_of_prods:
    number_of_occurrences = 0
    for prob in probabilities:
        if prob.startswith(str(lhs) + " "):
            number_of_occurrences += probabilities[prob]
    for prob in probabilities:
        if prob.startswith(str(lhs) + " "):
            probabilities[prob] = probabilities[prob] / number_of_occurrences

for index in range(len(productions_corpus)):
    prod = productions_corpus[index]
    productions_corpus[index] = ProbabilisticProduction(prod.lhs(), prod.rhs(), **{'prob': probabilities[str(prod)]})


def induce_pcfg2(start, productions):
    prods = [
        ProbabilisticProduction(prod.lhs(), prod.rhs(), prob=prod.prob())
        for prod in productions
    ]
    return PCFG(start, prods)


# this is the probability distribution for my corpus
fd_corpus = induce_pcfg2(Nonterminal('S'), productions_corpus)
