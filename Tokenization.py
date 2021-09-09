from util import *
class Tokenization():

    def naive(self, text):
        """
        Tokenization using a Naive Approach

        Parameters
        ----------
        arg1 : list
            A list of strings where each string is a single sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
        """

        tokenizedText = []

        for i in text:
            sentence_token = []
            #text_raw_token = re.split(' -|:|;|[\t\n]',i)
            text_raw_token = re.split(r'\s',i)
            #print('split text : ',text_raw_token)
            for j in text_raw_token :
                if (j == "") or (j in string.punctuation) :
                    continue
                sentence_token.append(j)
            #print('Punctuation removed :',sentence_token)
            tokenizedText.append(sentence_token)
        return tokenizedText

    def pennTreeBank(self, text):
        """
        Tokenization using the Penn Tree Bank Tokenizer

        Parameters
        ----------
        arg1 : list
            A list of strings where each string is a single sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
        """

        tokenizedText = []

        #Fill in code here
        p_t = nltk.tokenize.treebank.TreebankWordTokenizer()
        for i in text:
            sentence_token = []
            text_raw_token = p_t.tokenize(i)
            for j in text_raw_token :
                if (j == "") or (j in string.punctuation):
                    continue
                sentence_token.append(j)
            tokenizedText.append(sentence_token)
        return tokenizedText