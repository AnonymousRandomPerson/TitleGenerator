import logging
import os

import nltk
import string
from collections import Counter, defaultdict
from nltk.corpus import stopwords

import RAKE #to get: pip install python-rake
from RAKE import Rake
from summa import keywords, summarizer #to get: pip install summa
import textrank #to get: pip install git+git://github.com/davidadamojr/TextRank.git
import math
import heapq

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)
_hdlr = logging.StreamHandler()
_formatter = logging.Formatter('%(levelname)-8s - %(message)s')
_hdlr.setFormatter(_formatter)
_hdlr.setLevel(logging.DEBUG)
logger.addHandler(_hdlr)

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
        #Store equal-sized groups of the stemmed tokens (use is optional)
        self.splits = []

def main(useRake=False, useSummaTextRank=False, useTextRank=False):
    #Assumes file is in the program directory
    logger.info("In main\n")

    ##########################
    logger.info("------ Begin Processing ------")
    
    logger.info("Please enter file name of .txt file with extension (e.g., text.txt):")
    file_name = raw_input()#"sample_text.txt"
    logger.info("\t %s" % file_name)
    logger.info("Got file name")
    logger.info("Opening file")
    text_file = open(file_name)
    logger.info("Reading file")
    raw_text = text_file.read().lower()

    #Convert raw text to word tokens
    logger.info("Tokenizing")
    tokens = nltk.word_tokenize(raw_text.translate(None, string.punctuation))
    
    #Remove stopwords    
    logger.info("Removing stopwords")
    stop_words = set(stopwords.words('english'))
    #NOTE: we need to include some more stopwords, as 'english' doesn't contain some stopwords
    #      related to journal articles (e.g., "et" and "al" in "et al.")
    stop_words.update(['et','al']) 
    filtered_text = [word for word in tokens if word not in stop_words]
    
    #Create Corpus object for input text
    logger.info("Creating corpus object")
    input_text = Corpus(raw_text, filtered_text)
    
    logger.info("Filtered words to use")
    logger.info("\t %s" % input_text.word_tokens[:5])

    #NOTE: stopwords are removed before POS tags assigned, this could
    #      potentially degrade POS tagging performance - may want to 
    #      switch this order
    #Demonstrate functions
    logger.info("Getting POS tags")
    input_text.pos_tags = pos_tagger(input_text)
    logger.info("\t %s" % input_text.pos_tags[:5])
    
    logger.info("Getting stemmed words")
    input_text.stemmed_words = stem_tokens(input_text)
    logger.info("\t %s" % input_text.stemmed_words[:5])
    
    # split the stemmed words into ~equal-sized groups
    logger.info("Splitting the stemmed words into groups")
    #logger.info("There are %s words in this group" % len(input_text.stemmed_words))
    num_splits = 2
    input_text.splits = split_tokens(input_text, num_splits)
    #for s in input_text.splits:
    #    logger.info("%s %s\n\n" % (s,len(s)))

    logger.info("Getting word frequency and proximity")
    cutoff = 0.125
    if len(input_text.word_tokens) < 250:
        cutoff = 0.33
    input_text.word_freq_proximity = stems_frequency_proximity(input_text, cutoff)
    #logger.info("\t %s" % (input_text.word_freq_proximity[u'becom'],))

    logger.info("Mapping filtered words and their stemmed forms")
    input_text.filtered_word_and_bases = stems_and_bases(input_text)
    #logger.info("\t %s" % input_text.filtered_word_and_bases[u'becom'])
    
    logger.info("Mapping POS tags and words")
    input_text.pos_tag_and_words = pos_tags_and_words(input_text)
    #logger.info("\t %s" % input_text.pos_tag_and_words['NNS'][:5])

    logger.info("------ End Processing ------\n\n")
  
    ##########################

 
    if useRake:
        logger.info("------ Begin Rake ------")
        """More information at: https://github.com/fabianvf/python-rake"""

        stop_words_list = list(stop_words)
        r = Rake(RAKE.SmartStopList())#stop_words_list)
        sorted_keywords = r.run(input_text.raw_text)
        logger.info("Sorted keywords: %s" % sorted_keywords[:5])
        logger.info("------ End Rake ------\n\n")
 
    if useSummaTextRank:
        logger.info("------ Begin SummaTextRank ------")
        """More information at https://github.com/summanlp/textrank"""
        logger.info("Sentence(s) summary: %s" % summarizer.summarize(raw_text))
        logger.info("Keywords: %s" % keywords.keywords(raw_text))

        logger.info("------ End SummaTextRank ------\n\n")

    if useTextRank:
        logger.info("------ Begin TextRank ------")
        """More information at https://github.com/davidadamojr/TextRank"""
        
        logger.info("Sentence(s) summary: %s " % textrank.extract_sentences(raw_text))
        logger.info("Keywords: %s" % textrank.extract_key_phrases(raw_text))

        logger.info("------ End TextRank ------\n\n")

    ##########################
    
    logger.info("------ Begin Weighting ------")
   
    logger.info("Calculating word weights")
    input_text.word_weights = get_word_weights(input_text)
  
    logger.info("Printing word weights")
    weight_thresh = -1
    print_words_with_weight_above(weight_thresh, input_text.word_weights, input_text)

    logger.info("------ End Weighting ------\n\n")

    ##########################

    #logger.info("------ Begin Building ------")
 

    #logger.info("------ End Building ------\n\n")

    ##########################

    logger.info("Closing file")
    text_file.close()





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

#Split words in a corpus in num groups
def split_tokens(corpus, num):
    splits = []
    stemmed_words = corpus.stemmed_words
    split_length = len(stemmed_words)/num
    for i in range(num):
        start_idx = i*split_length
        if i == num-1: splits.append(stemmed_words[start_idx:]); break
        splits.append(stemmed_words[start_idx:start_idx+split_length])
    return splits

#Creates a dictionary of a corpus' stemmed words and their frequencies and proximity
def stems_frequency_proximity(corpus, cutoff=0.125):
    #Calculate frequency of stemmed words
    stemmed_freq = Counter(corpus.stemmed_words)
    
    #Generate word proximity calculations:
    #Determtine number of word tokens to approximate length of text
    text_length = len(corpus.word_tokens)
    
    #Create introduction and conclusion word ranges assuming that introduction takes up first
    #cutoff (default: 1/8) of text and conclusion falls in last cutoff (default: 1/8).
    introduction_cutoff = round(float(text_length) * cutoff)
    conclusion_cutoff = text_length - round(float(text_length) * cutoff)
    
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
    #NOTE: Some stems can have multiple bases, needed to change this function to address
    #      this fact.  Also, duplicates are now prevented from being added.
    stems_with_bases = defaultdict(list)
    for token in corpus.word_tokens:
        stem = porter_stemmer.stem(token)
        if token not in stems_with_bases[stem]:
            stems_with_bases[stem].append(token)
    return stems_with_bases

#Create dictionary of pos and list of words for given pos
def pos_tags_and_words(corpus):
    pos_and_words = defaultdict(list)
    for word,pos in corpus.pos_tags:
        #NOTE: this check was added
        if word not in pos_and_words[pos]:
            pos_and_words[pos].append(word)
    return pos_and_words

#Calculate the tfidf score for the words in each group of word stems.
#Returns a list of dictionaries.
def get_tfidf_weight(stem_groups):
    # elements are dictionaries of stem:tftdf score pairs, each dictionary 
    #    corresponds to group in stem_groups with the same index
    tfidf_group_weights = []
    for group in stem_groups:
        stemmed_freq = Counter(group)
        tot_freqs = sum(stemmed_freq.values())
        tfidf_weights = {}
        for word in group:
            freq = stemmed_freq[word]
            tf = float(freq)/len(group)
            num_splits_containing = sum(1 for stems in stem_groups if word in stems)
            value = float(len(stem_groups))/num_splits_containing
            idf = math.log(float(len(stem_groups))/num_splits_containing)
            tfidf = tf * idf
            tfidf_weights[word] = tfidf
        tfidf_group_weights.append(tfidf_weights.copy())
    return tfidf_group_weights

#Create dictionary of words and their associated weights
def get_word_weights(input_text):
    word_freq_prox = input_text.word_freq_proximity
    #get the maximum word frequency
    max_freq = max([freq_prox[0] for freq_prox in word_freq_prox.values()])
    
    #a list of dictionaries containing word:tridf score pairs
    word_tfidf_weights = get_tfidf_weight(input_text.splits)
    #get a list of lists of all the word weights for each split
    split_weights = [split_weight_dict.values() for split_weight_dict in word_tfidf_weights]
    #get the maximum tfidf score
    max_tfidf = max(weight for weights in split_weights for weight in weights)

    word_weight_dict = {}
    #weighted sum of word proximity, frequency, and (potentially) tfidf
    for word in word_freq_prox.keys():
        #find the highest tfidf score associated with the word
        highest_word_tfidf = -1
        for stem_tfidf_dict in word_tfidf_weights:
            if word in stem_tfidf_dict.keys():
                tfidf_weight = stem_tfidf_dict[word]
                if tfidf_weight > highest_word_tfidf:
                    highest_word_tfidf = tfidf_weight
        
        (freq, prox) = word_freq_prox[word]
        #higher weight for words near the beginning or end of the document
        prox_mult = 1 #how important proximity is (lower: less important)
        prox_weight = (1 - prox)*prox_mult
        
        #higher weight for words that occur more frequently
        freq_mult = 0.5 #how important proximity is (lower: less important)
        freq_weight = (float(freq)/max_freq)*freq_mult
        
        #NOTE: I left the weight of this component at 0 because it might not be
        #      useful - but didn't know if there is absolutely no use for this.
        #higher weight for words with higher tfidf score, normalize tfidf score
        #between 0 and tfidf_mult
        tfidf_mult = 0#.5
        tfidf = (highest_word_tfidf / max_tfidf)*tfidf_mult
        
        word_weight_dict[word] = prox_weight+freq_weight+tfidf 
    return word_weight_dict

#Output (to the console) up to 20 words with weight above the weight threshold 
#provided
def print_words_with_weight_above(weight_thresh, weights, input_text):
    pos = defaultdict(list)  
    word_and_weights = []
    for word in weights.keys():
        word_weight = weights[word]
        if word_weight > weight_thresh:
            actual_words = input_text.filtered_word_and_bases[word]
            for word,p in input_text.pos_tags:
                for actual_word in actual_words:
                    #0: word, 1: pos 
                    if word==actual_word and p not in pos[word]:
                        pos[word].append(p)
            #add the real word weight, the actual words (not the stems), and their
            #associated pos tags
            heapq.heappush(word_and_weights,(-word_weight,actual_words, \
                    [pos[actual_word] for actual_word in actual_words]))

    #print out up to 20 rows of weights, pos tags, and words
    for i in range(min([20,len(weights.keys())])):
        curr_word = heapq.heappop(word_and_weights)
        logger.info("\tWeight, pos, word: %s %s \t%s" % (-curr_word[0], curr_word[1], curr_word[2]))


if __name__ == "__main__":
    #Start the basic command line interface
    logger.info("------------ Starting ------------")
    main()
    #main(useRake=True,useSummaTextRank=True,useTextRank=True)
    logger.info("------------ Finished ------------")
