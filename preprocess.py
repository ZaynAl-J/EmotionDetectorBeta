#text preprocessing
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

def preprocess(line):
    ps = PorterStemmer()
    # Leave only characters from a to z
    # print("removing characters")
    review = re.sub('[^a-zA-Z]', ' ', line)
    # Lowertext
    review = review.lower()
    # Turn string into list of words
    review = review.split()
    # Apply Stemming
    # For example changing, changed, change --> chang or studying, studies, study --> studi

    # Delete stop words like I, and, or
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    # Turn list into sentences
    return " ".join(review)