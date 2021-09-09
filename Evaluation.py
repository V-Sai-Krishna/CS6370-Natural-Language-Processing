from util import *

class Evaluation():

    def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The precision value as a number between 0 and 1
        """

        precision = -1
        #print('Rank : ',k,' ',len(set(query_doc_IDs_ordered[0:k]).intersection(set(true_doc_IDs))))
        #Fill in code here
        precision = len(set(query_doc_IDs_ordered[0:k]).intersection(set(true_doc_IDs)))/k

#         if precision <0.09:
#         	print(query_id)
        return precision


    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean precision value as a number between 0 and 1
        """

        meanPrecision = -1

        #Fill in code here
        s=0
        for query in query_ids:
            rid=[]
            for q in qrels:
                if(int(q["query_num"])==query):
                    rid.append(int(q["id"]))
            s+=self.queryPrecision(doc_IDs_ordered[query_ids.index(query)],query,rid,k)
        meanPrecision=s/len(query_ids)

        return meanPrecision


    def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The recall value as a number between 0 and 1
        """

        recall = -1

        #Fill in code here
        if(len(true_doc_IDs)!=0):
            recall = len(set(query_doc_IDs_ordered[0:k]).intersection(set(true_doc_IDs)))/len(true_doc_IDs)

        return recall


    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean recall value as a number between 0 and 1
        """

        meanRecall = -1

        #Fill in code here
        s=0
        for query in query_ids:
            rid=[]
            for q in qrels:
                if(int(q["query_num"])==query):
                    rid.append(int(q["id"]))
            s+=self.queryRecall(doc_IDs_ordered[query_ids.index(query)],query,rid,k)
        meanRecall=s/len(query_ids)

        return meanRecall


    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The fscore value as a number between 0 and 1
        """

        fscore = -1

        #Fill in code here
        if(len(true_doc_IDs)!=0):
            p=self.queryPrecision(query_doc_IDs_ordered,query_id,true_doc_IDs,k)
            r=self.queryRecall(query_doc_IDs_ordered,query_id,true_doc_IDs,k)

            if p==0 and r==0 :
                fscore = 0
            else :	
                fscore=2*p*r/(p+r)

        return fscore


    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean fscore value as a number between 0 and 1
        """

        meanFscore = -1

        #Fill in code here
        s=0
        for query in query_ids:
            rid=[]
            for q in qrels:
                if(int(q["query_num"])==query):
                    rid.append(int(q["id"]))
            s+=self.queryFscore(doc_IDs_ordered[query_ids.index(query)],query,rid,k)
        meanFscore=s/len(query_ids)

        return meanFscore


    def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k,relevance_queries):
        """
        Computation of nDCG of the Information Retrieval System
        at given value of k for a single query

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The nDCG value as a number between 0 and 1
        """

        nDCG = -1

        #Fill in code here
        sorted_relevance_queries = []
        sorted_true_doc_IDs = []
        while(len(relevance_queries) != 0): 
            max_value = max(relevance_queries)
            max_index = relevance_queries.index(max_value) 
            sorted_relevance_queries.append(max_value)
            sorted_true_doc_IDs.append(true_doc_IDs[max_index])
            del relevance_queries[max_index]
            del true_doc_IDs[max_index]

        DCG = 0
        IDCG = 0 
        for i in range(min(k,len(sorted_relevance_queries))) : 
            if str(query_doc_IDs_ordered[i]) in sorted_true_doc_IDs : 
                DCG = DCG + (np.power(2,sorted_relevance_queries[sorted_true_doc_IDs.index(str(query_doc_IDs_ordered[i]))])-1)/(np.log2(i+2))
                #DCG = DCG + (sorted_relevance_queries[sorted_true_doc_IDs.index(str(query_doc_IDs_ordered[i]))])/(np.log2(i+2))
            else : 
                DCG = DCG + 1/(np.log2(i+2))
                #DCG = DCG	
            IDCG = IDCG + (np.power(2,sorted_relevance_queries[i])-1)/(np.log2(i+2))
            #IDCG = IDCG + (sorted_relevance_queries[i])/(np.log2(i+2))
        nDCG = DCG/IDCG 

        return nDCG


    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of nDCG of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries for which the documents are ordered
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The mean nDCG value as a number between 0 and 1
        """

        meanNDCG = -1

        #Fill in code here
        m_nDCG = 0 

        for i in range(len(doc_IDs_ordered)) :
            relevance_queries = []
            true_doc_IDs = []

            for j in qrels : 
                if (int(j["query_num"]) == query_ids[i]) : 
                    relevance_queries.append(5-j["position"])
                    true_doc_IDs.append(j["id"])

            m_nDCG = m_nDCG + self.queryNDCG(doc_IDs_ordered[i], query_ids[i], true_doc_IDs, k,relevance_queries)

        m_nDCG = m_nDCG/len(doc_IDs_ordered)

        meanNDCG = m_nDCG

        return meanNDCG


    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of average precision of the Information Retrieval System
        at a given value of k for a single query (the average of precision@i
        values for i such that the ith document is truly relevant)

        Parameters
        ----------
        arg1 : list
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        arg2 : int
            The ID of the query in question
        arg3 : list
            The list of documents relevant to the query (ground truth)
        arg4 : int
            The k value

        Returns
        -------
        float
            The average precision value as a number between 0 and 1
        """

        avgPrecision = -1

        #Fill in code here
        ap=0
        num=0
        den=0
        for id in query_doc_IDs_ordered[0:k]:
            den+=1
            if (id in true_doc_IDs):
                num+=1
                ap+=num/den
        avgPrecision = ap/len(true_doc_IDs)
        return avgPrecision


    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
        """
        Computation of MAP of the Information Retrieval System
        at given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        arg2 : list
            A list of IDs of the queries
        arg3 : list
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        arg4 : int
            The k value

        Returns
        -------
        float
            The MAP value as a number between 0 and 1
        """

        meanAveragePrecision = -1

        #Fill in code here
        s=0
        for query in query_ids:
            rid=[]
            for q in q_rels:
                if(int(q["query_num"])==query):
                    rid.append(int(q["id"]))
            s+=self.queryAveragePrecision(doc_IDs_ordered[query_ids.index(query)],query,rid,k)
        meanAveragePrecision=s/len(query_ids)

        return meanAveragePrecision