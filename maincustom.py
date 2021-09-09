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
    print("How many different variations of Search Engine do you wanna compare:")
    NS=int(input())
    labels=[]
    precisions=[]
    recalls=[]
    fscores=[]
    maps=[]
    ndcgs=[]
    for ns in range(NS):
        # Create an instance of the Search Engine
        loop=1
        while(loop==1):
            print("Press 0 to use TFIDF or Press 1 to use OKAPIBM25 for building index:")
            ind=int(input())
            if(ind==0):
                loop=0
                searchEngine = TFIDF(args)
            elif(ind==1):
                loop=0
                searchEngine = BM(args)
            else:
                print("Invalid!")
        loop=1
        while(loop==1):
            print("Press 1 to use WordNet else press 0:")
            pchoice=int(input())
            if((pchoice!=0) and (pchoice!=1)):
                print("Invalid!")
            else:
                loop=0
        
        loop=1
        while(loop==1):
            print("Press 1 to include Bigrams. Else Press 0:")
            bchoice=int(input())
            if((bchoice!=0) and (bchoice!=1)):
                print("Invalid!")
            else:
                loop=0
        loop=1
        while(loop==1):
            print("Press 1 to include LSA, else press 0:")
            lsachoice=int(input())
            if((lsachoice!=0) and (lsachoice!=1)):
                print("Invalid!")
            else:
                loop=0

        # Either handle query from user or evaluate on the complete dataset 
        if args.custom:
            searchEngine.handleCustomQuery(pchoice,bchoice,lsachoice)
        
        else:
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
        for ns in range(NS):
            plt.plot(range(1, 11), precisions[ns], label=labels[ns])
        plt.title("Precision")
        plt.xlabel("k")
        plt.legend()
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

        plt.subplot(1,5,2)
        for ns in range(NS):
            plt.plot(range(1, 11), recalls[ns], label=labels[ns])
        plt.title("Recall")
        plt.xlabel("k")
        plt.legend()
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


        plt.subplot(1,5,3)
        for ns in range(NS):
            plt.plot(range(1, 11), fscores[ns], label=labels[ns])
        plt.title("Fscore")
        plt.xlabel("k")
        plt.legend()
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


        plt.subplot(1,5,4)
        for ns in range(NS):
            plt.plot(range(1, 11), maps[ns], label=labels[ns])
        plt.title("MAP")
        plt.xlabel("k")
        plt.legend()
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)


        plt.subplot(1,5,5)
        for ns in range(NS):
            plt.plot(range(1, 11), ndcgs[ns], label=labels[ns])
        plt.title("nDCG")
        plt.xlabel("k")
        plt.legend()
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

        plt.savefig(args.out_folder + "customplot.png")

        
        if not os.path.exists('customcsvfolder'):
            os.makedirs('customcsvfolder')
        
        for j in range(NS):
            filename='customcsvfolder/'+labels[j]+".csv"
            with open(filename,'w',newline='') as file:
                writer=csv.writer(file)
                writer.writerow(["Precision","Recall","F-Score","MAP","nDCG"])
                for i in range(10):
                    writer.writerow([precisions[j][i],recalls[j][i],fscores[j][i],maps[j][i],ndcgs[j][i]])