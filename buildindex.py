from util import *
def buildIndex(docs, docIDs):

    word_list=list(set([word for doc in docs for word in doc]))
    Nd=len(docs)
    Nw=len(word_list)
    tf_matrix=np.zeros((Nd,Nw))
    idf_matrix = np.zeros((Nd,Nw))

    for i in range(len(docIDs)) : 
        doc = docs[i]
        if not len(doc) == 0 : 
            flatten_doc = list(set((doc)))
            for j in flatten_doc : 
                index_word = word_list.index(j)
                idf_matrix[:,index_word] = idf_matrix[:,index_word] + 1 

        for j in range(len(doc)) : 
            index_word = word_list.index(doc[j])
            tf_matrix[i,index_word]  = tf_matrix[i,index_word] + 1

    idf_matrix = -np.log(idf_matrix/len(docIDs))

    tfidf_matrix = np.transpose(tf_matrix * idf_matrix)
    idf_vector = idf_matrix[0,:].reshape((-1,1))
    index = [int(x) for x in docIDs]
    word_list=word_list
    
    return tfidf_matrix,idf_vector,index,word_list

