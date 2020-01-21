import nltk

sg = """
S -> NP VP | S NP | S PREP NP
VP -> IV | TV NP
NP -> SNNom SN | PNNom PN | SNDet SN | PNDet PN | 'John' | 'bread' | 'Mary' | 'They' | 'her' | 'She' | 'them' | 'Everybody' | 'it' | 'butter'
SNNom -> SNDet Adj
PNNom -> PNDet Adj
Adj -> 'heavy'
PREP -> 'to'
SNDet -> 'A' | 'The' | 'a'
PNDet -> 'many' | 'The' | 'Some'
SN -> 'boy' | 'chair' | 'book'
PN -> 'boys'
IV -> 'left'
TV -> 'eats' | 'loves' | 'love' | 'gave' | 'likes' | 'moves'
"""

g = nltk.CFG.fromstring(sg)

# Bottom-up  parser
sr_parser = nltk.ShiftReduceParser(g, trace=2)


# Parse sentences and observe the behavior of the parser
def parse_sentence(sent):
    tokens = sent.split()
    trees = sr_parser.parse(tokens)
    for tree in trees:
        print(tree)


'''
parse_sentence("John left")
# should be:
# (S (NP John) (VP (IV left)))
parse_sentence("John eats bread")
# should be:
# (S (NP John) (VP (TV eats) (NP bread)))
parse_sentence("John loves Mary")
parse_sentence("They love Mary")
parse_sentence("They love her")
parse_sentence("She loves them")
parse_sentence("Everybody loves John")
parse_sentence("A boy loves Mary")
parse_sentence("The boy loves Mary")
parse_sentence("Some boys love Mary")
parse_sentence("John gave Mary a heavy book")
'''
parse_sentence("John gave it to Mary")
'''
parse_sentence("John likes butter")
parse_sentence("John moves a chair")
'''
