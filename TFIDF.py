from util import *
from Process import Process 
from buildindex import buildIndex
from Evaluation import Evaluation

class TFIDF():

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
            doc_IDs_ordered = [] 
    
            tfidf,idf,index,wlist=buildIndex(pdocs,doc_ids)
            if(lsachoice==1):
                [U,S,V] = np.linalg.svd(tfidf, full_matrices=False, compute_uv=True, hermitian=False)
                a=600
                U1=U[:,0:a]
                S1=np.diag(S[0:a])
                V1=V[0:a,:]
                tfidf=np.matmul(np.matmul(U1,S1),V1)

            modDocs=np.sqrt(np.sum(tfidf**2 , 0)).reshape(-1,1)

            for i in range(len(pqueries)):
                Q=np.zeros((len(wlist),1))
                for word in pqueries[i]:
                    if (word in wlist):
                        Q[wlist.index(word)]+=1
                Q = Q*idf
                modQ=np.sqrt(np.sum(Q**2))
                cos_similarity = np.sum(tfidf*Q,0)/modQ
                for i in range(cos_similarity.shape[0]) : 
                    if not modDocs[i] == 0 :  
                        cos_similarity[i] =cos_similarity[i]/modDocs[i]
                    else : 
                        cos_similarity[i] = 0

                sortedq=[]

                index_copy = np.array(index.copy())
                
                while(len(cos_similarity)!=0):
                    sortedq.append(index_copy[np.argmax(cos_similarity)])
                    index_max = np.argmax(cos_similarity)
                    cos_similarity=np.delete(cos_similarity,index_max)
                    index_copy=np.delete(index_copy,index_max)
                doc_IDs_ordered.append(sortedq)

        elif(bchoice==1):
            bigram = models.Phrases(pdocs, min_count=1, threshold=0.01)
            bigram_mod = models.phrases.Phraser(bigram)
            pbidocs=[bigram_mod[doc] for doc in pdocs]
            pbiq=[bigram_mod[query] for query in pqueries]
            pdocs2=[list(set(pbidocs[i]) - set(pdocs[i])) for i in range(len(pdocs))]
            pqueries2=[list(set(pbiq[i]) - set(pqueries[i])) for i in range(len(pqueries))]
            doc_IDs_ordered = []
            tfidf,idf,index,wlist=buildIndex(pdocs,doc_ids)
            tfidf2,idf2,index2,wlist2=buildIndex(pdocs2,doc_ids)
            
            if(lsachoice==1):
                [U,S,V] = np.linalg.svd(tfidf, full_matrices=False, compute_uv=True, hermitian=False)
                a=600
                U1=U[:,0:a]
                S1=np.diag(S[0:a])
                V1=V[0:a,:]
                tfidf=np.matmul(np.matmul(U1,S1),V1)        
            
            modDocs=np.sqrt(np.sum(tfidf**2 , 0)).reshape(-1,1)
            modDocs2=np.sqrt(np.sum(tfidf2**2 , 0)).reshape(-1,1)
            for i in range(len(pqueries)):
                Q=np.zeros((len(wlist),1))
                for word in pqueries[i]:
                    if (word in wlist):
                        Q[wlist.index(word)]+=1
            #             Q+=comatrix2[:,wlist.index(word)].reshape(4265,1)
                Q = Q*idf
                modQ=np.sqrt(np.sum(Q**2))
                cos_similarity = np.sum(tfidf*Q,0)/modQ
                for j in range(cos_similarity.shape[0]) : 
                    if not modDocs[j] == 0 :  
                        cos_similarity[j] =cos_similarity[j]/modDocs[j]
                    else : 
                        cos_similarity[j] = 0

                Q2=np.zeros((len(wlist2),1))
                for word in pqueries2[i]:
                    if (word in wlist2):
                        Q2[wlist2.index(word)]+=1
            #             Q+=comatrix2[:,wlist.index(word)].reshape(4265,1)
                Q2 = Q2*idf2
                modQ2=np.sqrt(np.sum(Q2**2))
                cos_similarity2 = np.sum(tfidf2*Q2,0)/(modQ2+1e-8)
                for j in range(cos_similarity2.shape[0]) : 
                    if not modDocs[j] == 0 :  
                        cos_similarity2[j] =cos_similarity2[j]/(modDocs2[j])
                    else : 
                        cos_similarity2[j] = 0
                
                
                cos_similarity=0.6*cos_similarity+0.4*cos_similarity2
                
                sortedq=[]

                index_copy = np.array(index.copy())
                
                while(len(cos_similarity)!=0):
                    sortedq.append(index_copy[np.argmax(cos_similarity)])
                    index_max = np.argmax(cos_similarity)
                    cos_similarity=np.delete(cos_similarity,index_max)
                    index_copy=np.delete(index_copy,index_max)   
                doc_IDs_ordered.append(sortedq)



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
            doc_IDs_ordered = [] 
            # Build document index
            # Rank the documents for the query 

            tfidf,idf,index,wlist=buildIndex(pdocs,doc_ids)
            if(lsachoice==1):
                [U,S,V] = np.linalg.svd(tfidf, full_matrices=False, compute_uv=True, hermitian=False)
                a=600
                U1=U[:,0:a]
                S1=np.diag(S[0:a])
                V1=V[0:a,:]
                tfidf=np.matmul(np.matmul(U1,S1),V1)

            modDocs=np.sqrt(np.sum(tfidf**2 , 0)).reshape(-1,1)
            for i in range(len(pqueries)):
                Q=np.zeros((len(wlist),1))
                for word in pqueries[i]:
                    if (word in wlist):
                        Q[wlist.index(word)]+=1
                Q = Q*idf
                modQ=np.sqrt(np.sum(Q**2))
                cos_similarity = np.sum(tfidf*Q,0)/modQ
                for i in range(cos_similarity.shape[0]) : 
                    if not modDocs[i] == 0 :  
                        cos_similarity[i] =cos_similarity[i]/modDocs[i]
                    else : 
                        cos_similarity[i] = 0

                sortedq=[]

                index_copy = np.array(index.copy())
                
                while(len(cos_similarity)!=0):
                    sortedq.append(index_copy[np.argmax(cos_similarity)])
                    index_max = np.argmax(cos_similarity)
                    cos_similarity=np.delete(cos_similarity,index_max)
                    index_copy=np.delete(index_copy,index_max)
                doc_IDs_ordered.append(sortedq)


        elif(bchoice==1):
            bigram = models.Phrases(pdocs, min_count=1, threshold=0.01)
            bigram_mod = models.phrases.Phraser(bigram)
            pbidocs=[bigram_mod[doc] for doc in pdocs]
            pbiq=[bigram_mod[query] for query in pqueries]
            pdocs2=[list(set(pbidocs[i]) - set(pdocs[i])) for i in range(len(pdocs))]
            pqueries2=[list(set(pbiq[i]) - set(pqueries[i])) for i in range(len(pqueries))]
            doc_IDs_ordered = []
            tfidf,idf,index,wlist=buildIndex(pdocs,doc_ids)
            tfidf2,idf2,index2,wlist2=buildIndex(pdocs2,doc_ids)

            if(lsachoice==1):
                [U,S,V] = np.linalg.svd(tfidf, full_matrices=False, compute_uv=True, hermitian=False)
                a=600
                U1=U[:,0:a]
                S1=np.diag(S[0:a])
                V1=V[0:a,:]
                tfidf=np.matmul(np.matmul(U1,S1),V1)

            modDocs=np.sqrt(np.sum(tfidf**2 , 0)).reshape(-1,1)
            modDocs2=np.sqrt(np.sum(tfidf2**2 , 0)).reshape(-1,1)
            for i in range(len(pqueries)):
                Q=np.zeros((len(wlist),1))
                for word in pqueries[i]:
                    if (word in wlist):
                        Q[wlist.index(word)]+=1
            #             Q+=comatrix2[:,wlist.index(word)].reshape(4265,1)
                Q = Q*idf
                modQ=np.sqrt(np.sum(Q**2))
                cos_similarity = np.sum(tfidf*Q,0)/modQ
                for j in range(cos_similarity.shape[0]) : 
                    if not modDocs[j] == 0 :  
                        cos_similarity[j] =cos_similarity[j]/modDocs[j]
                    else : 
                        cos_similarity[j] = 0

                Q2=np.zeros((len(wlist2),1))
                for word in pqueries2[i]:
                    if (word in wlist2):
                        Q2[wlist2.index(word)]+=1
            #             Q+=comatrix2[:,wlist.index(word)].reshape(4265,1)
                Q2 = Q2*idf2
                modQ2=np.sqrt(np.sum(Q2**2))
                cos_similarity2 = np.sum(tfidf2*Q2,0)/(modQ2+1e-8)
                for j in range(cos_similarity2.shape[0]) : 
                    if not modDocs[j] == 0 :  
                        cos_similarity2[j] =cos_similarity2[j]/(modDocs2[j])
                    else : 
                        cos_similarity2[j] = 0
                
                
                cos_similarity=0.6*cos_similarity+0.4*cos_similarity2
                
                sortedq=[]

                index_copy = np.array(index.copy())
                
                while(len(cos_similarity)!=0):
                    sortedq.append(index_copy[np.argmax(cos_similarity)])
                    index_max = np.argmax(cos_similarity)
                    cos_similarity=np.delete(cos_similarity,index_max)
                    index_copy=np.delete(index_copy,index_max)   
                doc_IDs_ordered.append(sortedq)
 
        # Print the IDs of first five documents
        print("\nTop five document IDs : ")
        for id_ in doc_IDs_ordered[:5]:
            print(id_)