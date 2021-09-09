from util import *
from SentenceSegmentation import SentenceSegmentation
from Tokenization import Tokenization
from InflectionReduction import InflectionReduction
from StopwordRemoval import StopwordRemoval
tk = Tokenization()
ss = SentenceSegmentation()
ir = InflectionReduction()
sr = StopwordRemoval()


def reemovNestings(l,output): 
    for i in l: 
        if type(i) == list: 
            reemovNestings(i,output) 
        else: 
            output.append(i) 
    return output

def wnet(texts):
    addedtext=[]
    for i in range(len(texts)):
        line=texts[i].copy()
        for x in texts[i]:
            count=0
            for syn in wordnet.synsets(x):
                for l in syn.lemmas() :
                    if(count<1):
                        if(l.name() not in line):
                            line.append(l.name())
                            count+=1
        addedtext.append(line)
    return addedtext

def Process(docs,choice):
    if(choice==0):
        processeddocs=[]
        for doc in docs:
            d=ss.punkt(doc)
            bad_chars = ['|','+','=','_','-','~','`',';', ':', '!', "*","-","/",".","@","$","&","%",",",'(',')','[',']','{','}','<','>',"'",'"','1','2','3','4','5','6','7','8','9','0'] 
            for i in range(len(d)):
                for char in d[i]:
                    if char in bad_chars: 
                        d[i]=d[i].replace(char,' ')
            d1=tk.pennTreeBank(d)
            d2=sr.fromList(d1)
            o=[]
            d2=reemovNestings(d2,o)
            d3=ir.reduce(d2)
            d3=ir.reduce(d3)
            o=[]
            d5=reemovNestings(d3,o)
    #         for words in d5:
    #             if len(words) < 3:
    #                 d5.remove(words) 
            processeddocs.append(d5)

    elif(choice==1):
        processeddocs=[]
        for doc in docs:
            d=ss.punkt(doc)
            bad_chars = ['|','+','=','_','-','~','`',';', ':', '!', "*","-","/",".","@","$","&","%",",",'(',')','[',']','{','}','<','>',"'",'"','1','2','3','4','5','6','7','8','9','0'] 
            for i in range(len(d)):
                for char in d[i]:
                    if char in bad_chars: 
                        d[i]=d[i].replace(char,' ')
            d1=tk.pennTreeBank(d)
            d2=sr.fromList(d1)
            d3=wnet(d2)
            o=[]
            d3=reemovNestings(d3,o)           
            d3=ir.reduce(d3)
            d3=ir.reduce(d3)
            o=[]
            d5=reemovNestings(d3,o)
    #         for words in d5:
    #             if len(words) < 3:
    #                 d5.remove(words) 
            processeddocs.append(d5)
    return processeddocs
