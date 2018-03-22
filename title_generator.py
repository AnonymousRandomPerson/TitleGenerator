import nltk
import string
from collections import Counter, defaultdict
from nltk.corpus import stopwords

#Script to take input text file  and create appropriate titles

#Define stemmer for use in multiple functions
porter_stemmer = nltk.PorterStemmer()

class Corpus:
    def __init__(self, raw_text, word_tokens):
        #Store raw text of text file
        self.raw_text = raw_text
        #Store tokens of text in text file
        self.word_tokens = word_tokens
        #Store pos tags for tokens
        self.pos_tags = []
        #Store stemmed tokens 
        self.stemmed_words = []
        #Store dictionary of a corpus' stemmed words and their frequencies and proximity
        self.word_freq_proximity = {}
        #Store dictionary of filtered words with associated base words
        self.filtered_word_and_bases = {}
        #Store dictionary of POS tags with list of words for given pos
        self.pos_tag_and_words = {}

def main():
    #Assumes file is in the program directory
    file_name = raw_input('Please enter file name of .txt file with extension (i.e. text.txt):')
    text_file = open(file_name)
    raw_text = text_file.read().lower()
    
    #Convert raw text to word tokens
    tokens = nltk.word_tokenize(raw_text.translate(None, string.punctuation))
    
    #Remove stopwords    
    stop_words = set(stopwords.words('english'))
    filtered_text = [word for word in tokens if word not in stop_words]
    
    #Create Corpus object for input text
    input_text = Corpus(raw_text, filtered_text)
    
    #Demonstrate functions
    input_text.pos_tags = pos_tagger(input_text)
    input_text.stemmed_words = stem_tokens(input_text)
    input_text.word_freq_proximity = stems_frequency_proximity(input_text)
    input_text.filtered_word_and_bases = stems_and_bases(input_text)
    input_text.pos_tag_and_words = pos_tags_and_words(input_text)

#Create pos tags from tokenized text
def pos_tagger(corpus):
    pos = nltk.pos_tag(corpus.word_tokens)    
    return pos

#Stem words for a given corpus
def stem_tokens(corpus):
    stemmed_tokens = []
    for t in corpus.word_tokens:
        stemmed_tokens.append(porter_stemmer.stem(t))    
    return stemmed_tokens

#Creates a dictionary of a corpus' stemmed words and their frequencies and proximity
def stems_frequency_proximity(corpus):
    #Calculate frequency of stemmed words
    stemmed_freq = Counter(corpus.stemmed_words)
    
    #Generate word proximity calculations:
    #Determtine number of word tokens to approximate length of text
    text_length = len(corpus.word_tokens)
    
    #Create introduction and conclusion word ranges assuming that introduction takes up first
    #1/8 of text and conclusion falls in last 1/8.
    introduction_cutoff = round(float(text_length) * (.125))
    conclusion_cutoff = text_length - round(float(text_length) * (.125))
    
    #Create dictionary to hold stemmed words, frequency, and proximity value
    stemmed_word_freq = {}
    
    #Determine if index of word token is before introduction cutoff or after conclusion cutoff
    #Caclulate appropriate proximity score and add to dictionary
    for index, word in enumerate(corpus.word_tokens):
        proximity_base_score = index
        stem = porter_stemmer.stem(word)
        if stem not in stemmed_word_freq.keys():
            if proximity_base_score < introduction_cutoff:
                stemmed_word_freq[stem] = (stemmed_freq[stem], generate_word_weight(index, text_length))
            elif proximity_base_score > conclusion_cutoff:
                stemmed_word_freq[stem] = (stemmed_freq[stem], generate_word_weight(index, text_length))
            else:
                stemmed_word_freq[stem] = (stemmed_freq[stem], float(index)/text_length)
    return stemmed_word_freq

#Create word weight based on word's proximity and arbitrary weight factor
def generate_word_weight(base_index, text_length):
    #Calculate new weight based on given word index and length of text it belongs to
    #Will skew word weight to be lower if 'found' in introduction or conclusion
    weight_factor = .05
    word_weight_score = (float(base_index)/text_length) * weight_factor
    return word_weight_score

#Create dictionary of word stems with associated orginal word token
def stems_and_bases(corpus):
    stems_with_bases = {}
    for token in corpus.word_tokens:
        stems_with_bases[porter_stemmer.stem(token)] = token
    return stems_with_bases

#Create dictionary of pos and list of words for given pos
def pos_tags_and_words(corpus):
    pos_and_words = defaultdict(list)
    for word,pos in corpus.pos_tags:
        pos_and_words[pos].append(word)
    return pos_and_words

if __name__ == "__main__":
    #Start the basic command line interface
    main()
    