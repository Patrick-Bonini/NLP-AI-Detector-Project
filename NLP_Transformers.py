import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

nlp = spacy.load("en_core_web_trf")
doc = nlp("Apple shares rose on the news. Apple pie is delicious.")

class Categories:
    HUMAN = 'HUMAN'
    AI = 'AI'

textFromFile = open('training_data.txt', encoding="utf8")
text = textFromFile.read()
text = text.split('@')

train_x = [text[0], text[1], text[2], text[3], text[4], text[5], text[6], text[7], text[8], text[9], text[10], text[11], text[12]]
train_y = [Categories.AI, Categories.AI, Categories.HUMAN, Categories.HUMAN, Categories.HUMAN,
            Categories.AI, Categories.AI, Categories.AI, Categories.HUMAN, Categories.HUMAN, 
            Categories.AI, Categories.HUMAN, Categories.AI]

vectorizer = CountVectorizer(binary=True)
clf_svm = svm.SVC(kernel='linear')
train_x_vectors = vectorizer.fit_transform(train_x)
clf_svm.fit(train_x_vectors, train_y)

#Classifier
def classifier():
    while True:
        user_data = input(str('Paste your text here (one paragraph at a time):    '))
        test_x = vectorizer.transform([user_data])
        print(clf_svm.predict(test_x)) 

if __name__ == '__main__':
    classifier()