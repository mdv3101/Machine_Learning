from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
import operator


conf = SparkConf().setMaster("local").setAppName("SparkTFIDF")
sc = SparkContext(conf = conf)
rawData = sc.textFile("D:\DataScience\DataScience\subset-small.tsv")
fields = rawData.map(lambda x: x.split("\t"))
documents = fields.map(lambda x: x[3].split(" "))
documentNames = fields.map(lambda x: x[1])
hashingTF = HashingTF(100000)  
article_hash_value = hashingTF.transform(documents)
article_hash_value.cache()
st= "Abraham Lincoln Gettysburg"
wo = st.split()

idf = IDF(minDocFreq=2).fit(article_hash_value)
tfidf = idf.transform(article_hash_value)

def df_of_given_word (article_hash_value, word_hash):
    gtbTF = hashingTF.transform([word_hash])
    gtbHashValue1 = int(gtbTF.indices[0])
    gtbRelevance1 = tfidf.map(lambda x: x[gtbHashValue1] )
    y= [] 
    for z in gtbRelevance1.toLocalIterator():
        y.append(z)    
    return y
    
b = []
  
for i in range(len(wo)):
    c = []
    c= df_of_given_word(article_hash_value,wo[i])
    if i == 0:
        b = c
    else:
       b = map(operator.add, b,c) 
dn=[]
for z in documentNames.toLocalIterator():
        dn.append(z)
zipp= map(lambda x,y: (x,y),dn,b)   
zipp.sort(key=lambda tup: tup[1], reverse = True)   
print "Best document for " +st+  " is:"
print zipp[0]