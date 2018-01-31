import numpy as np
import pandas as pd

df = pd.read_csv('/home/arpit/learning/machine learning/quora/questions.csv')
print df.tail(),df.shape[0],df.shape[1]
df1 = pd.DataFrame()
df1['qid'] = df['qid1']
df1['questions']= df['question1']
df2 = pd.DataFrame()
df2['qid'] = df['qid2']
df2['questions'] = df['question2']
frame = [df1,df2]
df = pd.concat(frame)
print df.tail(),df.shape[0]