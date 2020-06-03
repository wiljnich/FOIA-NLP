import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer

df = pd.read_csv('data/foia_federal.csv', low_memory=False)

counter = Counter(df['org_id'].tolist())
agencies = {i[0]: idx for idx, i in enumerate(counter.most_common(50))}
df = df[df['org_id'].map(lambda x: x in agencies)]

requests = df['req'].tolist()
agency_list = [agencies[i] for i in df['org_id'].tolist()]
agency_list = np.array(agency_list)

count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(requests)

tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

train_x, test_x, train_y, test_y = train_test_split(x_train_tfidf, agency_list, test_size=0.3)

clf = MultinomialNB().fit(train_x, train_y)
y_score = clf.predict(test_x)

n_right = 0
for i in range(len(y_score)):
    if y_score[i] == test_y[i]:
        n_right += 1

print("Percentage of Requests Correctly Matched: %.0f%%" % ((n_right/float(len(test_y)) * 100)))