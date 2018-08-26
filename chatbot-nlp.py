import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


def clean_utterance(utterance):
    # Remove non alpha chars, convert charts to lowercase and split the string into a list
    utterance_words = re.sub('[^a-zA-Z]', ' ', utterance).lower().split()
    # Remove unnecessary words
    # Stemming - Only keep roots of words ex: loved -> love
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in utterance_words if word not in set(stopwords.words('english'))]
    # Join words back into a single string
    return ' '.join(stemmed_words)


def create_corpus(utterances):
    return [clean_utterance(utterance) for utterance in utterances]


class ChatbotNLP:
    def __init__(self, model_type=GaussianNB):
        self.cv = CountVectorizer()
        self.label_encoder = LabelEncoder()
        self.classifier = model_type()

    # Create a Bag of Words model, then trains the classifier using the given training data
    def train(self, training_data):
        X = self.cv.fit_transform(create_corpus(training_data['Utterance'])).toarray()
        y = self.label_encoder.fit_transform(training_data['Intent'])
        self.classifier.fit(X, y)

    def test(self, utterance):
        X = self.cv.transform([clean_utterance(utterance)]).toarray()
        # prediction = self.classifier.predict(X)
        prediction = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(prediction)[0]


dataset = pd.read_csv('utterances_and_intents.tsv', delimiter='\t')
nltk.download('stopwords')

nlp = ChatbotNLP()
nlp.train(dataset)

print(nlp.test("Hey, how you doing?"))
