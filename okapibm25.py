from util import *
def bm(docs,docIDs):
    word_list=list(set([word for doc in docs for word in doc]))
    Nd=len(docs)
    Nw=len(word_list)
    f_matrix=np.zeros((Nd,Nw))
    idf_matrix = np.zeros((Nw,1))
    D_matrix=np.zeros((len(docIDs),1))
    avgdl = 0
    for i in range(len(docIDs)) : 
        doc = docs[i]
        D_matrix[i] = len(doc)
        avgdl += len(doc)
        if not len(doc) == 0 : 
            flatten_doc = list(set((doc)))
            for j in flatten_doc : 
                index_word = word_list.index(j)
                idf_matrix[index_word] = idf_matrix[index_word] + 1 

        for j in range(len(doc)) : 
            index_word = word_list.index(doc[j])
            f_matrix[i,index_word]  = f_matrix[i,index_word] + 1

    idf_matrix = np.log((len(docs)-idf_matrix+0.5)/(idf_matrix+0.5)+1)

    index = [int(x) for x in docIDs]
    word_list=word_list
    avgdl = avgdl/len(docs)
    return f_matrix,idf_matrix,index,word_list,D_matrix,avgdl