import keras
# keras.__version__
print(keras.__version__)
import pandas as pd 
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import re 

# print("Emotion Detector")
# print("Creating Training, Validation, and Testing datasets:")

# # Create Training, Validation & Testing datasets
# train = pd.read_table('train.txt', delimiter=';', header=None, )
# val = pd.read_table('val.txt', delimiter=';', header=None, )
# test = pd.read_table('test.txt', delimiter=';', header=None, )

# # print("training \n", train)
# # print("validation \n", val)
# # print("testing \n", test)

# # Connect the datasets
# data = pd.concat([train, val, test])
# data.columns = ["text","label"]

# Save to file
# data.to_csv('data.csv')
# print(data)
# print(data.shape)
# Check if dataset has missing values:
#  data.isna().any(axis=1).sum()

# Text Preprocessing
# ps = PorterStemmer()

def preprocess(line):
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

def main():
    print("Emotion Detector")
    print("Creating Training, Validation, and Testing datasets:")

    # Create Training, Validation & Testing datasets
    train = pd.read_table('train.txt', delimiter=';', header=None, )
    val = pd.read_table('val.txt', delimiter=';', header=None, )
    test = pd.read_table('test.txt', delimiter=';', header=None, )

    # print("training \n", train)
    # print("validation \n", val)
    # print("testing \n", test)

    # Connect the datasets
    data = pd.concat([train, val, test])
    data.columns = ["text","label"]

    # Save to file
    data.to_csv('data.csv')

    # Apply text preprocessing:
    data['text'] = data['text'].apply(lambda x: preprocess(x))
    data.to_csv('Preprocessed.csv')

main()