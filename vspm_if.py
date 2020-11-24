import glob
import nltk
import sys
import os
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
from collections import Counter
import numpy as np
from collections import OrderedDict

def wordList_removePuncs(doc_dict):
    wordList = []
    stop = stopwords.words('english') + list(string.punctuation) + ['\n']
    for doc in doc_dict.values():
        for word in word_tokenize(doc.lower().strip()): 
            if not word in stop:
                wordList.append(word)
    return wordList

def give_path(fld_path):                             
    dic = {}
    path = 'D:/src/InforRetri/du_lich'

    for r,d,files in os.walk(path): 
        for file in files:
            name = file.split('/')[-1]
            file_path = path + '/' + file
            print(name)
            with open(file_path, 'r',encoding="utf8", errors='ignore') as f:
                data = f.read()
            dic[name] = data
    return dic

def termFrequencyInDoc(vocab, doc_dict):
    tf_docs = {}
    for doc_id in doc_dict.keys():
        tf_docs[doc_id] = {}
    
    for word in vocab:
        for doc_id,doc in doc_dict.items():
            tf_docs[doc_id][word] = doc.count(word)
    return tf_docs

def wordDocFre(vocab, doc_dict):
    df = {}
    for word in vocab:
        frq = 0
        for doc in doc_dict.values():
            if word in word_tokenize(doc.lower().strip()):
                frq = frq + 1
        df[word] = frq
    return df

def inverseDocFre(vocab,doc_fre,length):
    idf= {} 
    for word in vocab:     
        idf[word] = np.log2((length+1) / doc_fre[word])
    return idf

def tfidf(vocab,tf,idf_scr,doc_dict):
    tf_idf_scr = {}
    for doc_id in doc_dict.keys():
        tf_idf_scr[doc_id] = {}
    for word in vocab:
        for doc_id,doc in doc_dict.items():
            tf_idf_scr[doc_id][word] = tf[doc_id][word] * idf_scr[word]
    return tf_idf_scr

def vectorSpaceModel(query, doc_dict,tfidf_scr):
    query_vocab = []
    for word in query.split():
        if word not in query_vocab:
            query_vocab.append(word)

    query_wc = {}
    for word in query_vocab:
        query_wc[word] = query.lower().split().count(word)
    
    relevance_scores = {}
    for doc_id in doc_dict.keys():
        score = 0
        for word in query_vocab:
            score += query_wc[word] * tfidf_scr[doc_id][word]
        relevance_scores[doc_id] = score
    sorted_value = OrderedDict(sorted(relevance_scores.items(), key=lambda x: x[1], reverse = True))
    top_5 = {k: sorted_value[k] for k in list(sorted_value)[:5]}
    return top_5#print(lst_contents)

# Bước 1: Đọc các file văn bản
path = './du_lich/*.txt'
docs = give_path(path)                        
#print(docs)

M = len(docs)
print(M)                                       

# Bước 2: xây dưng tập vocab của các văn bản
'''
w_List = wordList_removePuncs(docs)           

for word in w_List:
    if word[0] != " ` " or word[0] != " ' ":
path = './vocab.txt'
f = open(path, 'r', encoding="utf8")
str = f.read()
words = str.replace('"', '').replace('.', '').replace("'","").replace("`","").split()
print(words)
f = open("./vocab.txt", 'w', encoding="utf8")
for word in words:
    f.write('{} '.format(word))
'''

# Bước 3: xây dựng tập từ điển của các băn bản
'''
vocab = list(set(w_List))
print(vocab)                     

# Bước 4: tính TF (term frequency)
tf_dict = termFrequencyInDoc(vocab, docs)
#print(tf_dict)

# Bước 5: tính DF (document Frequency)     
df_dict = wordDocFre(vocab, docs)    
#print(df_dict)

# Bước 6: tính IDF         
idf_dict = inverseDocFre(vocab,df_dict,M)
#print(idf_dict)

# Bước 7: tính TF_IDF     
tf_idf = tfidf(vocab,tf_dict,idf_dict,docs)   
#print(tf_idf)

# Bước 8: nhập query 
search_query = sys.argv[1]


# Bước 9: thực hiện truy vấn với vector space model
res = vectorSpaceModel(search_query, docs, tf_idf)
print(res)
'''