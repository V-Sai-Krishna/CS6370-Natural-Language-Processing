from util import *
from Process import Process 
from buildindex import buildIndex
from Evaluation import Evaluation
from TFIDF import TFIDF
from BM import BM
# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")


if __name__ == "__main__":

    # Create an argument parser
    parser = argparse.ArgumentParser(description='main.py')

    # Tunable parameters as external arguments
    parser.add_argument('-dataset', default = "cranfield/", 
                        help = "Path to the dataset folder")
    parser.add_argument('-out_folder', default = "output/", 
                        help = "Path to output folder")
    parser.add_argument('-segmenter', default = "punkt",
                        help = "Sentence Segmenter Type [naive|punkt]")
    parser.add_argument('-tokenizer',  default = "ptb",
                        help = "Tokenizer Type [naive|ptb]")
    parser.add_argument('-custom', action = "store_true", 
                        help = "Take custom query as input")

    # Parse the input arguments
    args = parser.parse_args()
    print(args)
    labels=[]
    precisions=[]
    recalls=[]
    fscores=[]
    maps=[]
    ndcgs=[]

    # Baseline model
    ind=0
    pchoice=0
    bchoice=0
    lsachoice=0
    searchEngine=TFIDF(args)
    p,r,f,m,n=searchEngine.evaluateDataset(pchoice,bchoice,lsachoice)
    precisions.append(p)
    recalls.append(r)
    fscores.append(f)
    maps.append(m)
    ndcgs.append(n)
    label=''
    if(ind==0):
        label=label+'TFIDF'
    else:
        label=label+'BM25'
    if(pchoice==1):
        label+='+Wordnet'
    if(bchoice==1):
        label+='+Bigram'
    if(lsachoice==1):
        label+='+LSA'
    labels.append(label)


    # TF-IDF+LSA
    ind=0
    pchoice=0
    bchoice=0
    lsachoice=1
    searchEngine=TFIDF(args)
    p,r,f,m,n=searchEngine.evaluateDataset(pchoice,bchoice,lsachoice)
    precisions.append(p)
    recalls.append(r)
    fscores.append(f)
    maps.append(m)
    ndcgs.append(n)
    label=''
    if(ind==0):
        label=label+'TFIDF'
    else:
        label=label+'BM25'
    if(pchoice==1):
        label+='+Wordnet'
    if(bchoice==1):
        label+='+Bigram'
    if(lsachoice==1):
        label+='+LSA'
    labels.append(label)


    # TF-IDF+Wordnet
    ind=0
    pchoice=1
    bchoice=0
    lsachoice=0
    searchEngine=TFIDF(args)
    p,r,f,m,n=searchEngine.evaluateDataset(pchoice,bchoice,lsachoice)
    precisions.append(p)
    recalls.append(r)
    fscores.append(f)
    maps.append(m)
    ndcgs.append(n)
    label=''
    if(ind==0):
        label=label+'TFIDF'
    else:
        label=label+'BM25'
    if(pchoice==1):
        label+='+Wordnet'
    if(bchoice==1):
        label+='+Bigram'
    if(lsachoice==1):
        label+='+LSA'
    labels.append(label)

    # BM25
    ind=1
    pchoice=0
    bchoice=0
    lsachoice=0
    searchEngine=BM(args)
    p,r,f,m,n=searchEngine.evaluateDataset(pchoice,bchoice,lsachoice)
    precisions.append(p)
    recalls.append(r)
    fscores.append(f)
    maps.append(m)
    ndcgs.append(n)
    label=''
    if(ind==0):
        label=label+'TFIDF'
    else:
        label=label+'BM25'
    if(pchoice==1):
        label+='+Wordnet'
    if(bchoice==1):
        label+='+Bigram'
    if(lsachoice==1):
        label+='+LSA'
    labels.append(label)

    #Bigram
    ind=0
    pchoice=0
    bchoice=1
    lsachoice=0
    searchEngine=TFIDF(args)
    p,r,f,m,n=searchEngine.evaluateDataset(pchoice,bchoice,lsachoice)
    precisions.append(p)
    recalls.append(r)
    fscores.append(f)
    maps.append(m)
    ndcgs.append(n)
    label=''
    if(ind==0):
        label=label+'TFIDF'
    else:
        label=label+'BM25'
    if(pchoice==1):
        label+='+Wordnet'
    if(bchoice==1):
        label+='+Bigram'
    if(lsachoice==1):
        label+='+LSA'
    labels.append(label)


    if(args.custom==0):
        plt.figure(1)
        plt.subplots(1,5,figsize=(20,5))
        plt.subplot(1,5,1)
        plt.plot(range(1, 11), precisions[0], label=labels[0])
        plt.plot(range(1, 11), precisions[1], label=labels[1])
        plt.plot(range(1, 11), precisions[2], label=labels[2])
        plt.plot(range(1, 11), precisions[3], label=labels[3])
        plt.plot(range(1, 11), precisions[4], label=labels[4])
        plt.title("Precision")
        plt.xlabel("k")
        plt.legend()
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

        plt.subplot(1,5,2)
        plt.plot(range(1, 11), recalls[0], label=labels[0])
        plt.plot(range(1, 11), recalls[1], label=labels[1])
        plt.plot(range(1, 11), recalls[2], label=labels[2])
        plt.plot(range(1, 11), recalls[3], label=labels[3])
        plt.plot(range(1, 11), recalls[4], label=labels[4])
        plt.title("Recall")
        plt.xlabel("k")
        plt.legend()
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


        plt.subplot(1,5,3)
        plt.plot(range(1, 11), fscores[0], label=labels[0])
        plt.plot(range(1, 11), fscores[1], label=labels[1])
        plt.plot(range(1, 11), fscores[2], label=labels[2])
        plt.plot(range(1, 11), fscores[3], label=labels[3])
        plt.plot(range(1, 11), fscores[4], label=labels[4])
        plt.title("Fscore")
        plt.xlabel("k")
        plt.legend()
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


        plt.subplot(1,5,4)
        plt.plot(range(1, 11), maps[0], label=labels[0])
        plt.plot(range(1, 11), maps[1], label=labels[1])
        plt.plot(range(1, 11), maps[2], label=labels[2])
        plt.plot(range(1, 11), maps[3], label=labels[3])
        plt.plot(range(1, 11), maps[4], label=labels[4])
        plt.title("MAP")
        plt.xlabel("k")
        plt.legend()
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


        plt.subplot(1,5,5)
        plt.plot(range(1, 11), ndcgs[0], label=labels[0])
        plt.plot(range(1, 11), ndcgs[1], label=labels[1])
        plt.plot(range(1, 11), ndcgs[2], label=labels[2])
        plt.plot(range(1, 11), ndcgs[3], label=labels[3])
        plt.plot(range(1, 11), ndcgs[4], label=labels[4])
        plt.title("nDCG")
        plt.xlabel("k")
        plt.legend()
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

        plt.savefig(args.out_folder + "Experiment1.png")


        
        if not os.path.exists('expt1csvfolder'):
            os.makedirs('expt1csvfolder')
        
        for j in range(5):
            filename='expt1csvfolder/'+labels[j]+".csv"
            with open(filename,'w',newline='') as file:
                writer=csv.writer(file)
                writer.writerow(["Precision","Recall","F-Score","MAP","nDCG"])
                for i in range(10):
                    writer.writerow([precisions[j][i],recalls[j][i],fscores[j][i],maps[j][i],ndcgs[j][i]])
