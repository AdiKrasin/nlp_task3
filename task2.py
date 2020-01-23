from nltk.grammar import Nonterminal
from nltk.grammar import toy_pcfg2
from nltk.probability import DictionaryProbDist
from nltk.tree import Tree
import os


'''
productions = toy_pcfg2.productions()
# Get all productions with LHS=NP
np_productions = toy_pcfg2.productions(Nonterminal('NP'))

dict = {}
for pr in np_productions: dict[pr.rhs()] = pr.prob()
np_probDist = DictionaryProbDist(dict)

# Each time you call, you get a random sample
print(np_probDist.generate())
'''


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
            genereted_nts[index] = nts_into_ts(Tree(old_nt, t))
        return genereted_nts

    productions = grammar.productions()
    dic = dict()
    for pr in productions: dic[pr.rhs()] = pr.prob()
    productions_probDist = DictionaryProbDist(dic)
    genereted = productions_probDist.generate()
    genereted = Tree('S', [genereted[0], genereted[1]])
    return nts_into_ts(genereted)


file_content = ""
for i in range(1000):
    res_tree = pcfg_generate(toy_pcfg2)
    file_content += str(res_tree) + "\n"

if os.path.exists(".\\toy_pcfg2.gen"):
    os.remove(".\\toy_pcfg2.gen")
else:
    with open(".\\toy_pcfg2.gen", "w+") as f:
        f.write(file_content)
