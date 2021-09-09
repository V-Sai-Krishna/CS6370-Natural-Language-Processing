# EE17B035-EE17B047: NLP Final Project README File

This folder contains the files used for the Project, involving building a search engine application. Note that this code works for Python 3 only. 

The files and folders present are:
1) Folder Cranfield: Contains the Cranfield Documents, queries and the ranking of documents for each queries.

2) expt1csvfolder: precision@k, recall@k, f-score@k, nDCG@k and the MAP@k values obtained from Experiment 1 shown in the report are stored as csv files in this folder.
3) expt2csvfolder: precision@k, recall@k, f-score@k, nDCG@k and the MAP@k values obtained from Experiment 2 shown in the report are stored as csv files in this folder.
4) expt3csvfolder: precision@k, recall@k, f-score@k, nDCG@k and the MAP@k values obtained from Experiment 3 shown in the report are stored as csv files in this folder.

5) output folder: Graphs of experiments are stored in this folder

6) buildindex.py: Same as that in Assignment 2 
7) SentenceSegmentation.py: Same as that in Assignment 2 
8) StopwordRemoval.py: Same as that in Assignment 2 
9) Tokenization.py: Same as that in Assignment 2 
10) util.py: Same as that in Assignment 2 
11) inflectionReduction.py: Same as that in Assignment 2 
12) Evaluation.py: Same as that in Assignment 2 
13) informationRetrieval.py;: Same as that in Assignment 2 


14) Process.py: Processed the documents (Sentence Segmentation, Stopword Removal, etc). Takes an argument called pchoice. If pchoice==1, wordnet will be used for query expansion.
15) okapibm25.py: Similar to buildindex.py. However, this python script creates index so that Okapi BM25 scoring metric can be used.
16) TFIDF.py: Creates a Search Engine that uses the buildindex.py to create index and uses TF-IDF scoring metric to rank documents.
17) BM.py: Creates a Search Engine that uses the okapibm25.py to create index and uses Okapi BM25 scoring metric to rank documents.


****************************************************** The scripts that you will run **********************************************************

18) main1.py: To replicate the experiments shown on section 6.1 of the report run this file. Output graph can be found in the outputs folder named 
              "Experiment 1.png". The Precision, Recall, etc values for each IR system used in this expt can be found in the folder "expt1csvfolder".

17) main2.py: To replicate the experiments shown on section 6.2 of the report run this file. Output graph can be found in the outputs folder named 
              "Experiment 2.png". The Precision, Recall, etc values for each IR system used in this expt can be found in the folder "expt2csvfolder".

17) main3.py: To replicate the experiments shown on section 6.3 of the report run this file. Output graph can be found in the outputs folder named 
              "Experiment 3.png". The Precision, Recall, etc values for each IR system used in this expt can be found in the folder "expt3csvfolder".

17) maincustom.py: To create and run custom IR systems run this file. Output graph can be found in the outputs folder named "customplot.png". 
                   The Precision, Recall, etc values for each IR system used in this expt can be found in the folder "customcsvfolder".



Once again, run only the following files:
1) main1.py
2) main2.py
3) main3.py
4) maincustom.py

To run main1/main2/main3/maincustom.py as before with the appropriate arguments (Similar to Assignment 2).
Usage Example: main1.py [-custom] [-dataset DATASET FOLDER] [-out_folder OUTPUT FOLDER]
               [-segmenter SEGMENTER TYPE (naive|punkt)] [-tokenizer TOKENIZER TYPE (naive|ptb)] 

When the -custom flag is passed, the system will take a query from the user as input. For example:
> python main.py -custom
> Enter query below
> Papers on Aerodynamics
This will print the IDs of the five most relevant documents to the query to standard output.


> main1.py, main2.py and main3.py don't take arguments than the ones mentioned above.

> maincustom.py is a user interactive program. It will let you create your own IR system (Ex: TF-IDF+Bigram or BM25+LSA+Wordnet, etc). Just follow the
instructions shown on the screen and you are good to go.

Disclaimer: Bigrams and LSA will take time to run (Since they need to compute SVD). Hence, be patient if you create those while running maincustom.py

