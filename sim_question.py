import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv('/home/arpit/learning/machine learning/quora/questions.csv')
# print df.tail(),df.shape[0],df.shape[1]
#check for null element
print df.isnull().any().count()
df1 = pd.DataFrame()
# df1['qid'] = df['qid1']
df1['questions']= df['question1']
df2 = pd.DataFrame()
# df2['qid'] = df['qid2']
df2['questions'] = df['question2']
frame = [df1,df2]
#add columns to get all data under single column
df = pd.concat(frame)
# print df.tail(),df.shape[0]
#drop null values
df=df.dropna()
# string_sim = string_sim.toarray()
string_sim = df.as_matrix()
print df.isnull().any()
# print string_sim[:10]
#make a tfidf matrix
tfidf = TfidfVectorizer(stop_words = 'english')
y = tfidf.fit_transform(df['questions'][:1000].values.astype('U'))
y_array = y.toarray()
#applying cosine similarity
sim_arry1 = cosine_similarity(y_array)
# print sim_arry1[:3]
l=-1
for i in sim_arry1:
	k=-1
	l=l+1
	for j in i:
		k=k+1
		if(j>0.8):
			if(string_sim[l]!=string_sim[k]):
				print j,string_sim[l],string_sim[k]

