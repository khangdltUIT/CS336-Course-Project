import pickle
from pyvi import ViTokenizer, ViPosTagger
import numpy as np
def load_model():
    file_path = ["./model.pkl", "idf_vector.pkl", "./word.pkl"]
    ### Load Model
    with open(file_path[0], 'rb') as f:
        model = pickle.load(f)
    ### Load Idf_vector
    with open(file_path[1], 'rb') as f:
        idf_vector = pickle.load(f)
    ### Load Word
    with open(file_path[2], 'rb') as f:
        word = pickle.load(f)
    
    return model, idf_vector, word

def preprocessing_query(word):
    dict_path = './dictionary_stopwords.txt'
    stopword_dict = open(dict_path, "r", encoding="utf8")
    dict = stopword_dict.read()
    word = ViTokenizer.tokenize(word)
    prew = ''
    for w in word.lower().split():
        if w not in dict:
            prew += ( w + ' ')
    return prew

#Scalar
def Scalar(model,words, idf_vector,query):
    content_query=query.split()
    words_query=list(set(content_query))
    vector_query=[]
    res=[]
    try:
        for i in words_query:
            tf=content_query.count(i)/len(content_query)
            idf=idf_vector[words.index(i)]
            vector_query.append(tf*idf)
        vector_query=np.array(vector_query)

        docs_set=set()
        for i in words_query:
            docs_set.update(model[i].keys())
        docs_set=list(docs_set)
        for i in docs_set:
            vt=list()
            for j in words_query:
                vt.append(model[j].setdefault(i,0))
            vt=np.array(vt)
            SProduct=vt@vector_query
            res.append((i,SProduct))
        res.sort(reverse=True,key=lambda x: x[1])
        return res[:5]
    except:
        return None

model, idf_vector, words = load_model()
query = "Nên du lịch ở đâu vào tháng 11"
print("QUERY: ", query )
query = preprocessing_query(query)
print("After processing", query)
print("------------------------------------------------\n Kết quả tìm kiếm:")
res = Scalar(model,words,idf_vector,query)
for doc in res:
    print(doc[0])
    path = "./du_lich/jUAD-"+str(doc[0])+'.txt'
    with open(path, 'r', encoding="utf8") as f:
        full_doc = f.read()
    print(full_doc)
    print("...................................................")