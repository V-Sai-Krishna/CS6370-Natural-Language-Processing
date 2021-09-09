import numpy as np
from sklearn.utils.extmath import randomized_svd
from gensim import corpora, models, similarities
from sys import version_info
import json
import matplotlib.pyplot as plt
import re
import nltk.data
import string
from nltk.corpus import stopwords, wordnet
from sklearn import metrics
import argparse
import os
import csv
