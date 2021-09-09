from util import *

class StopwordRemoval():

    def fromList(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence with stopwords removed
        """

        stopwordRemovedText = None
        stop_words = set(stopwords.words('english')) 
        stopwordRemovedText = [[j for j in i if not j in stop_words] for i in text]


        return stopwordRemovedText