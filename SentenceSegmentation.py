from util import *

class SentenceSegmentation():

    def naive(self, text):
        """
        Sentence Segmentation using a Naive Approach

        Parameters
        ----------
        arg1 : str
            A string (a bunch of sentences)

        Returns
        -------
        list
            A list of strings where each string is a single sentence
        """

        segmentedText = []

        #Fill in code here
        partial_segmented = text.split('.')

        for j in partial_segmented : 
            j_split = j.split('?')
            for k in j_split :
                k_split = k.split('!')
                [segmentedText.append(l.strip()) for l in k_split if not l.strip() == ""] 

        return segmentedText

    def punkt(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : str
            A string (a bunch of sentences)

        Returns
        -------
        list
            A list of strings where each strin is a single sentence
        """

        #Fill in code here
        tokenizer_punkt = nltk.data.load('tokenizers/punkt/english.pickle')
        segmentedText = tokenizer_punkt.tokenize(text.strip())

        return segmentedText