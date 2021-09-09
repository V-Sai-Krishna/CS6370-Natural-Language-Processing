from util import *
from Process import Process 
from okapibm25 import bm
from Evaluation import Evaluation
class BM():

    def __init__(self, args):
        self.args = args
        self.evaluator = Evaluation()

    def evaluateDataset(self,pchoice,bchoice,lsachoice):
        """
        - preprocesses the queries and documents, stores in output folder
        - invokes the IR system
        - evaluates precision, recall, fscore, nDCG and MAP 
            for all queries in the Cranfield dataset
        - produces graphs of the evaluation metrics in the output folder
        """

        # Read queries
        queries_json = json.load(open(self.args.dataset + "cran_queries.json", 'r'))[:]
        query_ids, queries = [item["query number"] for item in queries_json], \
                                [item["query"] for item in queries_json]
        # Process queries 
        pqueries = Process(queries,pchoice)


        # Read documents
        docs_json = json.load(open(self.args.dataset + "cran_docs.json", 'r'))[:]
        doc_ids, docs = [item["id"] for item in docs_json], \
                                [item["body"] for item in docs_json]
        # Process documents
        pdocs = Process(docs,0)

        if(bchoice==0):
            # Build document index
            # Rank the documents for each query
            k=1.5
            b=0.75
            doc_IDs_ordered = [] 
    
            f,idf,index,wlist,D,ad=bm(pdocs,doc_ids)
            if(lsachoice==1):
                [U,S,V] = np.linalg.svd(f, full_matrices=False, compute_uv=True, hermitian=False)
                a=1150
                U1=U[:,0:a]
                S1=np.diag(S[0:a])
                V1=V[0:a,:]
                f=np.matmul(np.matmul(U1,S1),V1)

            for i in range(len(pqueries)):
                similarity = np.zeros((len(pdocs),1))
                for word in pqueries[i]:
                    if (word in wlist):
                        similarity+=idf[wlist.index(word)]*(f[:,wlist.index(word)].reshape(1400,1)*(k+1)/(f[:,wlist.index(word)].reshape(1400,1)+k*(1-b+b*D/ad)))
                sortedq=[]

                index_copy = np.array(index.copy())
                
                while(len(similarity)!=0):
                    sortedq.append(index_copy[np.argmax(similarity)])
                    index_max = np.argmax(similarity)
                    similarity=np.delete(similarity,index_max)
                    index_copy=np.delete(index_copy,index_max)
                doc_IDs_ordered.append(sortedq)
        
        elif(bchoice==1):
            bigram = models.Phrases(pdocs, min_count=1, threshold=0.01)
            bigram_mod = models.phrases.Phraser(bigram)
            pbidocs=[bigram_mod[doc] for doc in pdocs]
            pbiq=[bigram_mod[query] for query in pqueries]
            pdocs2=[list(set(pbidocs[i]) - set(pdocs[i])) for i in range(len(pdocs))]
            pqueries2=[list(set(pbiq[i]) - set(pqueries[i])) for i in range(len(pqueries))]
            k=1.5
            b=0.75
            f,idf,index,wlist,D,ad=bm(pdocs,doc_ids)
            f2,idf2,index2,wlist2,D2,ad2=bm(pdocs2,doc_ids) 
            doc_IDs_ordered = []
            if(lsachoice==1):
                [U,S,V] = np.linalg.svd(f, full_matrices=False, compute_uv=True, hermitian=False)
                a=1150
                U1=U[:,0:a]
                S1=np.diag(S[0:a])
                V1=V[0:a,:]
                f=np.matmul(np.matmul(U1,S1),V1)  

            for i in range(len(pqueries)):
                similarity = np.zeros((len(pdocs),1))
                for word in pqueries[i]:
                    if (word in wlist):
                        similarity+=idf[wlist.index(word)]*(f[:,wlist.index(word)].reshape(1400,1)*(k+1)/(f[:,wlist.index(word)].reshape(1400,1)+k*(1-b+b*D/ad)))
                
                similarity2 = np.zeros((len(pdocs2),1))
                for word in pqueries2[i]:
                    if (word in wlist2):
                        similarity2+=idf2[wlist2.index(word)]*(f2[:,wlist2.index(word)].reshape(1400,1)*(k+1)/(f2[:,wlist2.index(word)].reshape(1400,1)+k*(1-b+b*D2/ad2)))   
            
                similarity=0.6*similarity+0.4*similarity2
                sortedq=[]

                index_copy = np.array(index.copy())
                
                while(len(similarity)!=0):
                    sortedq.append(index_copy[np.argmax(similarity)])
                    index_max = np.argmax(similarity)
                    similarity=np.delete(similarity,index_max)
                    index_copy=np.delete(index_copy,index_max)
                doc_IDs_ordered.append(sortedq)


        print("Index built")
        # Read relevance judements
        qrels = json.load(open(self.args.dataset + "cran_qrels.json", 'r'))[:]

        # Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
        precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
        for k in range(1,11):
            precision = self.evaluator.meanPrecision(
                doc_IDs_ordered, query_ids, qrels, k)
            precisions.append(precision)
            recall = self.evaluator.meanRecall(
                doc_IDs_ordered, query_ids, qrels, k)
            recalls.append(recall)
            fscore = self.evaluator.meanFscore(
                doc_IDs_ordered, query_ids, qrels, k)
            fscores.append(fscore)
            print("Precision, Recall and F-score @ " +  
                str(k) + " : " + str(precision) + ", " + str(recall) + 
                ", " + str(fscore))
            MAP = self.evaluator.meanAveragePrecision(
                doc_IDs_ordered, query_ids, qrels, k)
            MAPs.append(MAP)
            nDCG = self.evaluator.meanNDCG(
                doc_IDs_ordered, query_ids, qrels, k)
            nDCGs.append(nDCG)
            print("MAP, nDCG @ " +  
                str(k) + " : " + str(MAP) + ", " + str(nDCG))

        return precisions,recalls,fscores, MAPs, nDCGs

    def handleCustomQuery(self,pchoice,bchoice,lsachoice):
        """
        Take a custom query as input and return top five relevant documents
        """

        #Get query
        print("Enter query below")
        query = input()
        # Process documents
        pqueries = Process([query],pchoice)

        # Read documents
        docs_json = json.load(open(self.args.dataset + "cran_docs.json", 'r'))[:]
        doc_ids, docs = [item["id"] for item in docs_json], \
                            [item["body"] for item in docs_json]
        # Process documents
        pdocs = Process(docs,0)
        
        if(bchoice==0):
            # Build document index
            # Rank the documents for each query
            k=1.5
            b=0.75
            doc_IDs_ordered = [] 
    
            f,idf,index,wlist,D,ad=bm(pdocs,doc_ids)
            if(lsachoice==1):
                [U,S,V] = np.linalg.svd(f, full_matrices=False, compute_uv=True, hermitian=False)
                a=600
                U1=U[:,0:a]
                S1=np.diag(S[0:a])
                V1=V[0:a,:]
                f=np.matmul(np.matmul(U1,S1),V1)

            for i in range(len(pqueries)):
                similarity = np.zeros((len(pdocs),1))
                for word in pqueries[i]:
                    if (word in wlist):
                        similarity+=idf[wlist.index(word)]*(f[:,wlist.index(word)].reshape(1400,1)*(k+1)/(f[:,wlist.index(word)].reshape(1400,1)+k*(1-b+b*D/ad)))
                sortedq=[]

                index_copy = np.array(index.copy())
                
                while(len(similarity)!=0):
                    sortedq.append(index_copy[np.argmax(similarity)])
                    index_max = np.argmax(similarity)
                    similarity=np.delete(similarity,index_max)
                    index_copy=np.delete(index_copy,index_max)
                doc_IDs_ordered.append(sortedq)
        
        elif(bchoice==1):
            bigram = models.Phrases(pdocs, min_count=1, threshold=0.1)
            bigram_mod = models.phrases.Phraser(bigram)
            pbidocs=[bigram_mod[doc] for doc in pdocs]
            pbiq=[bigram_mod[query] for query in pqueries]
            pdocs2=[pdocs[i] + list(set(pbidocs[i]) - set(pdocs[i])) for i in range(len(pdocs))]
            pqueries2=[pqueries[i] + list(set(pbiq[i]) - set(pqueries[i])) for i in range(len(pqueries))]
            pdocs=pdocs2
            pqueries=pqueries2
        
            doc_IDs_ordered = [] 
            # Build document index
            # Rank the documents for each query
            k=1.5
            b=0.75

            f,idf,index,wlist,D,ad=bm(pdocs,doc_ids)
            if(lsachoice==1):
                [U,S,V] = np.linalg.svd(f, full_matrices=False, compute_uv=True, hermitian=False)
                a=600
                U1=U[:,0:a]
                S1=np.diag(S[0:a])
                V1=V[0:a,:]
                f=np.matmul(np.matmul(U1,S1),V1)

            for i in range(len(pqueries)):
                similarity = np.zeros((len(pdocs),1))
                for word in pqueries[i]:
                    if (word in wlist):
                        similarity+=idf[wlist.index(word)]*(f[:,wlist.index(word)].reshape(1400,1)*(k+1)/(f[:,wlist.index(word)].reshape(1400,1)+k*(1-b+b*D/ad)))
                sortedq=[]

                index_copy = np.array(index.copy())
                
                while(len(similarity)!=0):
                    sortedq.append(index_copy[np.argmax(similarity)])
                    index_max = np.argmax(similarity)
                    similarity=np.delete(similarity,index_max)
                    index_copy=np.delete(index_copy,index_max)
                doc_IDs_ordered.append(sortedq)
        
        # Print the IDs of first five documents
        print("\nTop five document IDs : ")
        for id_ in doc_IDs_ordered[:5]:
            print(id_)