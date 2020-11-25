import re
# file địa chỉ ban đầu
f = open("./dictionary_stopwords.txt", "r+", encoding="utf8" )
dic = f.read()
words = dic.replace(' ','_').split()
# file địa chỉ ghi vào
f = open("./pro_dictionary.txt", "w", encoding="utf8" )
for word in words:
    f.write("{}\n".format(word))
    
    