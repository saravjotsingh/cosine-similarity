import spacy
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
import re

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))


# read the test from file
file1 = open("data.txt","r")
text = file1.read();
file1.close()

def removeStopWords(text):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

    return ' '.join(filtered_sentence)
    
def cleanText(text):
    cleanString = re.sub('\W+',' ', text )

    return cleanString

def getCosineSimilarity(vect1, vect2):

    return 1 - distance.cosine(vect1, vect2)

def getKeywordVector(keyword):
    
    return nlp(keyword).vector

def getSimilarWords(keyword,allKeywords):

    keywordVector = getKeywordVector(keyword)

    similarity_list = []
    for token in allKeywords:
        similarity_list.append((token,getCosineSimilarity(keywordVector,token.vector)))

    similarity_list = sorted(similarity_list, key=lambda item: -item[1])
    similarity_list = similarity_list[:30]
    top_similar_words = [item[0].text for item in similarity_list]
    top_similar_words = top_similar_words[:3]
    top_similar_words = list(set(top_similar_words))
    return top_similar_words
    
 

clean_text = removeStopWords(text)
removed_sw_text = cleanText(clean_text)
allKeywords = nlp(removed_sw_text)

#Get the similar keyword according to provided text
print(getSimilarWords('safely',allKeywords))



# remove special characters and space


