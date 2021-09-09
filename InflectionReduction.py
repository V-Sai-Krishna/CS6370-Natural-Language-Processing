from util import *

class InflectionReduction:

    def reduce(self, text):
        """
        Stemming/Lemmatization

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of
            stemmed/lemmatized tokens representing a sentence
        """
        stemmer = nltk.stem.porter.PorterStemmer()
        reducedText = [stemmer.stem(j) for j in text]

#         lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
#         reducedText = [[lemmatizer.lemmatize(token) for token in doc] for doc in text]
        #Fill in code here

        return reducedText