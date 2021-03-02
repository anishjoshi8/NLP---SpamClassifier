nltk.download('all')
import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

messages = pd.read_csv('SMSSpamCollection', sep='\t', names=['label','message'])

corpus =[]
# sentences = nltk.sent_tokenize(messages)
wordLemmatize = WordNetLemmatizer()

for i in range(len(messages)):
  review = re.sub('[^a-zA-Z]',' ',messages['message'][i])
  review = review.lower()
  review = review.split()
  review = [wordLemmatize.lemmatize(w) for w in review if not w in stopwords.words('english')]
  review = ' '.join(review)
  corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train, y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score
confusion = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)