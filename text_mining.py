import pandas as pd
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#import nltk 
#from nltk.stem import WordNetLemmatizer
#nltk.download()
#import numpy as np
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer



import re
#from nltk.tokenize import RegexpTokenizer
#from nltk.corpus import stopwords
#from sklearn.feature_extraction.text import TfidVectorizer
porter = PorterStemmer()
remove_regex = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
given_word_list=['research', 'data', 'mining', 'analytics', 'data mining', 'machine learning', 'deep learning']
given_word_list = [porter.stem(word) for word in given_word_list]
given_word_list = [word.lower() for word in given_word_list]

def pre_processing(input_file_location):
    fd=open(input_file_location)
    doc=fd.read()
    
    cleantext = re.sub(remove_regex, '', doc)

    sentences = sent_tokenize(cleantext)

    tokens = word_tokenize(str(sentences))
    words = [word for word in tokens if word.isalpha()]

    stop_words = stopwords.words('english')
    words = [w for w in words if not w in stop_words]
    fd.close()
    porter = PorterStemmer()
    words = [porter.stem(word) for word in words]
    words = [word.lower() for word in words]
    string=words#' '.join(words)
    return string

def count_freq(doc_list,words_list):
    req_word_freq={}
    wordfreq = [doc_list.count(c) for c in doc_list]
    doc_freq_dictionary= dict(list(zip(doc_list,wordfreq)))
    #print (doc_freq_dictionary)
    for word in words_list:
        if(len(word.split())>1):
            wor=word.split()
            try:
                if(doc_freq_dictionary[wor[0]]<doc_freq_dictionary[wor[1]]):
                    try:
                        req_word_freq[word]=doc_freq_dictionary[wor[0]]
                    except:
                        req_word_freq[word]=0
                else:
                    try:
                        req_word_freq[word]=doc_freq_dictionary[wor[1]]
                    except:
                        req_word_freq[word]=0
            except:
                req_word_freq[word]=0
        else:
            try:
                req_word_freq[word]=doc_freq_dictionary[word]
            except:
                req_word_freq[word]=0
    return req_word_freq


f1=open('input_files/write1.txt','w')
f2=open('input_files/write2.txt','w')

doc1=pre_processing('input_files/view-source_cis.csuohio.edu__sschung_.txt')
doc1_dict=count_freq(doc1,given_word_list)
f1.write(str(doc1_dict))

#print(doc1_dict)
doc2=pre_processing('input_files/view-source_https___en.wikipedia.org_wiki_Data_mining.txt')
doc2_dict=(count_freq(doc2,given_word_list))
#print(doc2_dict)
f1.write(str(doc2_dict))

doc3=pre_processing('input_files/view-source_https___en.wikipedia.org_wiki_Data_mining#Data_mining.txt')
doc3_dict=(count_freq(doc3,given_word_list))
#print(doc3_dict)
f1.write(str(doc3_dict))

doc4=pre_processing('input_files/view-source_https___en.wikipedia.org_wiki_Engineering.txt')
doc4_dict=(count_freq(doc4,given_word_list))
#print(doc4_dict)
f1.write(str(doc4_dict))

doc5=pre_processing('input_files/view-source_https___my.clevelandclinic.org_research.txt')
doc5_dict=(count_freq(doc5,given_word_list))
#print(doc5_dict)
f1.write(str(doc5_dict))

doc6=pre_processing('input_files/view-source_https___www.edx.org_course_data-science-machine-learning.txt')
doc6_dict=(count_freq(doc5,given_word_list))
#print(doc6_dict)
f1.write(str(doc6_dict))

f1.close()

data_dict = [doc1_dict,doc2_dict,doc3_dict,doc4_dict,doc5_dict,doc6_dict]
# Create DictVectorizer object
f2.write(str(data_dict))
f2.close()


dictvectorizer = DictVectorizer(sparse=False)

# Convert dictionary into feature matrix
features = dictvectorizer.fit_transform(data_dict)

print("Part1 of assignment")

print(dictvectorizer.get_feature_names())

print(features)

print(features.shape)




from sklearn.feature_extraction.text import TfidfTransformer
tfidfTran = TfidfTransformer(norm="l2")
tfidfTran.fit(features)
print (tfidfTran.idf_)

print("Part2 of assignment Cosine measure")

tfidf_matrix = tfidfTran.transform(features)
cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
print(cos_similarity_matrix)

print("Part3 of assignment Analysis and Discussion of Problems")

print("Cosine similarity is a metric used to determine how similar two entities are irrespective of their size.\nIt measures the cosine of the angle between two vectors projected in a multi-dimensional space. If 'a' and 'b' are two vectors, cosine equation gives the angle between the two.")

print("The result from computing the similarity of Item A to Item B is the same as computing the similarity of Item B to Item A.")

print("Which 2 docs are most similar in terms of 7 given topics?  ans Data_mining.txt,Data_mining#Data_mining.txt are similar")


