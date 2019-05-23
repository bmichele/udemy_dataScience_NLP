###################################################################
# Define function to build Binary BOW and Count BOW from sentence #
###################################################################

import numpy as np

class VectorizerBOW():
    def __init__(self, method = 'count'):
        self.method = method
        self.word_to_index = {}
        self.dim = None
    def fit(self, corpus: list):
        '''
        Argument must be a list of lists, each sublist corresponding to a tokenized sentence.
        '''
        _vocab = set([el for sent in corpus for el in sent])
        self.word_to_index = {w: i for i, w in enumerate(_vocab)}
        self.dim = len(self.word_to_index)
        pass
    def transform(self, corpus: list) -> np.array:
        _out = []
        for sent in corpus:
            _vec = np.zeros(self.dim, dtype = int)
            if self.method == 'binary':
                for w in set(sent):
                    _vec[self.word_to_index[w]] = 1
                _out.append(_vec)
            elif self.method == 'count':
                for w in sent:
                    _vec[self.word_to_index[w]] += 1
                _out.append(_vec)
            else:
                print('Invalid method')
                pass
        return np.array(_out)


## Testing
vectorizer = VectorizerBOW(method='count')
vectorizer.fit(['ciao prova test'.split()])
vectorizer.transform(['prova prova test'.split()])