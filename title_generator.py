# coding=utf-8

import argparse
from collections import Counter, defaultdict
import heapq
import logging
import math
import os
import random
import re
import string

#pip install nltk numpy python-rake summa git+git://github.com/davidadamojr/TextRank.git
import nltk
from nltk.corpus import stopwords
import numpy as np
import RAKE
from RAKE import Rake
from summa import keywords, summarizer
import textrank

#POS tags are listed at https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html.

#POS tag for conjunctions.
CONJUNCTION = 'CC'
#POS tag for articles, quantity adjectives.
DETERMINER = 'DT'
#POS tag for prepositions.
PREPOSITION = 'IN'
#POS tag for personal pronouns.
PRONOUN_PERSONAL = 'PRP'
#POS tag for possessive pronouns.
PRONOUN_POSSESSIVE = 'PRP$'
#POS tag for regular adjectives.
ADJECTIVE = 'JJ'
#POS tag for comparative adjectives.
ADJECTIVE_COMPARATIVE = 'JJR'
#POS tag for superlative adjectives.
ADJECTIVE_SUPERLATIVE = 'JJS'
#POS tag for singular nouns.
NOUN = 'NN'
#POS tag for plural nouns.
NOUN_PLURAL = 'NNS'
#POS tag for proper nouns.
NOUN_PROPER = 'NNP'
#POS tag for proper plural nouns.
NOUN_PROPER_PLURAL = 'NNPS'
#POS tag for regular adverbs.
ADVERB = 'RB'
#POS tag for compartive adverbs.
ADVERB_COMPARATIVE = 'RBR'
#POS tag for superlative adverbs.
ADVERB_SUPERLATIVE = 'RBS'
#POS tag for 'to'.
TO = 'TO'
#POS tag for present participles and gerunds.
VERB_PRESENT_PARTICIPLE = 'VBG'
#Dummy POS tag for manually set words.
DEFAULT_TAG = 'DEFAULT'

#All of the possible noun tags.
NOUNS = (NOUN, NOUN_PLURAL, NOUN_PROPER, NOUN_PROPER_PLURAL)
ADJECTIVES = (ADJECTIVE, ADJECTIVE_COMPARATIVE, ADJECTIVE_SUPERLATIVE)
ADVERBS = (ADVERB, ADVERB_COMPARATIVE, ADVERB_SUPERLATIVE)

#A list of all part-of-speech templates for making words.
#Each POS template is a list of symbols, which can be any of the following:
# - POS tag string. The word will always have this POS tag.
# - Tuple of POS tag strings. The word will be any of the POS tags in the tuple.
# - Lowercase word. The word will be hard-coded as this word.
#Also note that adjectives and adverbs will be inserted sometimes and ignored other times, based on their probability constants below.
POS_TEMPLATES = [
    [VERB_PRESENT_PARTICIPLE, ADJECTIVES, NOUNS], #"Retitling Scientific Texts"
    [DETERMINER, ADJECTIVES, NOUNS],
    [DETERMINER, ADJECTIVES, NOUNS, 'of', ADJECTIVES, NOUNS], #"The Unreasonable Effectiveness of Mathematics"
    [ADJECTIVES, NOUNS, 'in', ADJECTIVES, NOUNS],
    [DETERMINER, ADJECTIVES, NOUNS, 'in', ADJECTIVES, NOUNS],
    [ADJECTIVE, NOUN],
    [ADJECTIVES, NOUNS, CONJUNCTION, ADJECTIVES, NOUNS], #"Word Weighting and Title Generation"
    [ADJECTIVES, NOUNS, ':', DETERMINER, ADJECTIVES, NOUNS], #"Artificial Intelligence: A Modern Approach"
    ['Towards', VERB_PRESENT_PARTICIPLE, ADJECTIVES, NOUNS],
    [DETERMINER, ADJECTIVES, NOUNS, 'for', NOUNS], #"A Data-centric Architecture for Search"
    [ADJECTIVES, NOUNS, 'for', VERB_PRESENT_PARTICIPLE, ADJECTIVES, NOUNS], #"Empirical Methods for Evaluating Dialogue Systems"
    [NOUNS, ',', DETERMINER, ADJECTIVES, NOUNS], #"GUS, A Driven System"
    [ADJECTIVES, NOUNS, 'for', NOUNS, ':', DETERMINER, ADJECTIVES, NOUNS] #"Dialogue Systems for Surveys: the Rate-a-course System"
]

#Parts of speech that allow the use of stopwords.
POS_USE_STOPWORDS = {CONJUNCTION, DETERMINER, PREPOSITION}
#Parts of speech that should not be capitalized in the title.
POS_NOT_CAPITAL = {CONJUNCTION, DETERMINER, PREPOSITION, PRONOUN_PERSONAL, PRONOUN_POSSESSIVE, TO}

#The level of logging for displaying debug information to the console.
LOGGER_LEVEL = logging.ERROR

#The number of titles to generate for the given text.
NUM_TITLES = 10

#A set of additional stopwords to add to the stopwords list beyond the default stopwords set.
ADDITIONAL_STOPWORDS = {'et', 'al'}

#A list of all vowels.
VOWELS = {'a', 'e', 'i', 'o', 'u'}

#Punctutation that is used in POS templates.
PUNCTUATION = {':',','}

#The chance of an adjective being used when encountered in a POS template.
ADJECTIVE_CHANCE = 0.5
#The chance of an adverb being used when encountered in a POS template.
ADVERB_CHANCE = 0.2


logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(LOGGER_LEVEL)
_hdlr = logging.StreamHandler()
_formatter = logging.Formatter('%(levelname)-8s - %(message)s')
_hdlr.setFormatter(_formatter)
_hdlr.setLevel(LOGGER_LEVEL)
logger.addHandler(_hdlr)

#Script to take input text file  and create appropriate titles

#Define stemmer for use in multiple functions
porter_stemmer = nltk.PorterStemmer()

class Corpus:
    def __init__(self, raw_text, all_tokens, filtered_tokens):
        #Store raw text of text file
        self.raw_text = raw_text
        #Store tokens of text in text file
        self.all_tokens = all_tokens
        #Store filtered tokens of text in text file when excluding stopwords
        self.filtered_tokens = filtered_tokens
        #Store pos tags for tokens
        self.pos_tags = []
        #Parts of speech that are used in the given text
        self.used_pos = {}
        #Store stemmed tokens
        self.stemmed_words = []
        #Store dictionary of a corpus' stemmed words and their frequencies and proximity
        self.word_freq_proximity = {}
        #Store list of stems of words in the introduction and/or conclusion
        self.intro_conc_stems = []
        #Store dictionary of filtered words with associated base words
        self.filtered_word_and_bases = {}
        #Store dictionary of filtered base words with associated words
        self.filtered_bases_and_words = {}
        #Store dictionary of POS tags with list of words for given pos
        self.pos_tag_and_words = {}
        #Store equal-sized groups of the stemmed tokens (use is optional)
        self.splits = []
        #Store stopwords that are ignored when weighting.
        self.stop_words = set()

def main(file_name, use_rake=False, use_summa_text_rank=False, use_text_rank=False):
    #Assumes file is in the program directory
    logger.info("In main\n")

    ##########################
    logger.info("------ Begin Processing ------")

    file_name = args.file_name
    if not file_name:
        print "Please enter file name of .txt file with extension (e.g., text.txt):"
        file_name = raw_input()#"sample_text.txt"
    logger.info("\t %s" % file_name)
    logger.info("Got file name")

    titles_ranked = generate_titles(file_name, use_rake, use_summa_text_rank, use_text_rank)

    logger.info("------ Begin Print ------")

    print_titles(titles_ranked)

    logger.info("------ End Print ------\n\n")

def generate_titles(file_name, use_rake=False, use_summa_text_rank=False, use_text_rank=False):
    logger.info("Opening file")
    text_file = open(file_name)
    logger.info("Reading file")
    raw_text = text_file.read().lower()
    # Remove Unicode characters.
    raw_text = raw_text.decode('unicode_escape').encode('ascii','ignore')

    #Convert raw text to word tokens
    logger.info("Tokenizing")
    tokens = nltk.word_tokenize(raw_text.translate(None, string.punctuation))

    #Remove stopwords
    logger.info("Removing stopwords")
    stop_words = set(stopwords.words('english'))
    #NOTE: we need to include some more stopwords, as 'english' doesn't contain some stopwords
    #      related to journal articles (e.g., "et" and "al" in "et al.")
    stop_words.update(ADDITIONAL_STOPWORDS)
    filtered_text = [word for word in tokens if word not in stop_words]

    #Create Corpus object for input text
    logger.info("Creating corpus object")
    input_text = Corpus(raw_text, tokens, filtered_text)
    input_text.stop_words = stop_words

    logger.info("Filtered words to use")
    logger.info("\t %s" % input_text.filtered_tokens[:5])

    #NOTE: stopwords are removed before POS tags assigned, this could
    #      potentially degrade POS tagging performance - may want to
    #      switch this order
    #Demonstrate functions
    logger.info("Getting POS tags")
    input_text.pos_tags = pos_tagger(input_text)
    logger.info("\t %s" % input_text.pos_tags[:5])

    logger.info("Finding all used parts of speech.")
    input_text.used_pos = set([tag[1] for tag in input_text.pos_tags])
    logger.info(input_text.used_pos)

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
    if len(input_text.filtered_tokens) < 250:
        cutoff = 0.33
    input_text.word_freq_proximity = stems_frequency_proximity(input_text, cutoff)
    #logger.info("\t %s" % (input_text.word_freq_proximity[u'becom'],))

    logger.info("Mapping filtered words and their stemmed forms")
    input_text.filtered_word_and_bases, input_text.filtered_bases_and_words = stems_and_bases(input_text)
    #logger.info("\t %s" % input_text.filtered_word_and_bases[u'becom'])

    logger.info("Mapping POS tags and words")
    input_text.pos_tag_and_words = pos_tags_and_words(input_text)
    #logger.info("\t %s" % input_text.pos_tag_and_words['NNS'][:5])

    logger.info("------ End Processing ------\n\n")

    ##########################


    if use_rake:
        logger.info("------ Begin Rake ------")
        """More information at: https://github.com/fabianvf/python-rake"""

        r = Rake(RAKE.SmartStopList())#stop_words_list)
        sorted_keywords = r.run(input_text.raw_text)
        logger.info("Sorted keywords: %s" % sorted_keywords[:5])
        logger.info("------ End Rake ------\n\n")

    if use_summa_text_rank:
        logger.info("------ Begin SummaTextRank ------")
        """More information at https://github.com/summanlp/textrank"""
        logger.info("Sentence(s) summary: %s" % summarizer.summarize(raw_text))
        logger.info("Keywords: %s" % keywords.keywords(raw_text))

        logger.info("------ End SummaTextRank ------\n\n")

    if use_text_rank:
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

    logger.info("------ Begin Building ------")

    titles = build_titles(input_text)

    logger.info("------ End Building ------\n\n")

    ##########################

    logger.info("Closing file")
    text_file.close()

    ##########################

    logger.info("------ Begin Ranking ------")

    #NOTE: the scores denote the title rankings relative to one another
    #      1 denotes the "best" title (highest absolute sum of word weights)
    #      and 0 denotes the "worst" of the presented titles
    titles_ranked = order_titles(titles, input_text)

    logger.info("------ End Ranking ------\n\n")

    ##########################

    return titles_ranked

#Create pos tags from tokenized text
def pos_tagger(corpus):
    pos = nltk.pos_tag(corpus.all_tokens)
    return pos

#Stem words for a given corpus
def stem_tokens(corpus):
    stemmed_tokens = []
    for t in corpus.filtered_tokens:
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
    text_length = len(corpus.filtered_tokens)

    #Create introduction and conclusion word ranges assuming that introduction takes up first
    #cutoff (default: 1/8) of text and conclusion falls in last cutoff (default: 1/8).
    introduction_cutoff = round(float(text_length) * cutoff)
    conclusion_cutoff = text_length - round(float(text_length) * cutoff)

    #Create dictionary to hold stemmed words, frequency, and proximity value
    stemmed_word_freq = {}

    #Determine if index of word token is before introduction cutoff or after conclusion cutoff
    #Caclulate appropriate proximity score and add to dictionary
    for index, word in enumerate(corpus.filtered_tokens):
        proximity_base_score = index
        stem = porter_stemmer.stem(word)
        if stem not in stemmed_word_freq.keys():
            if proximity_base_score < introduction_cutoff:
                stemmed_word_freq[stem] = (stemmed_freq[stem], generate_word_weight(index, text_length))
                if stem not in corpus.intro_conc_stems:
                    corpus.intro_conc_stems.append(stem)
            elif proximity_base_score > conclusion_cutoff:
                stemmed_word_freq[stem] = (stemmed_freq[stem], generate_word_weight(index, text_length))
                if stem not in corpus.intro_conc_stems:
                    corpus.intro_conc_stems.append(stem)
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

#Create dictionary of word stems with associated orginal word token, along with its reverse dictionary
def stems_and_bases(corpus):
    #NOTE: Some stems can have multiple bases, needed to change this function to address
    #      this fact.  Also, duplicates are now prevented from being added.
    stems_with_bases = defaultdict(list)
    bases_with_stems = defaultdict(str)
    for token in corpus.filtered_tokens:
        stem = porter_stemmer.stem(token)
        if token not in stems_with_bases[stem]:
            stems_with_bases[stem].append(token)
        bases_with_stems[token] = stem

    return stems_with_bases, bases_with_stems

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
        tfidf_weights = {}
        for word in group:
            freq = stemmed_freq[word]
            tf = float(freq)/len(group)
            num_splits_containing = sum(1 for stems in stem_groups if word in stems)
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
    for stem in word_freq_prox.keys():
        #find the highest tfidf score associated with the stem
        highest_word_tfidf = -1
        for stem_tfidf_dict in word_tfidf_weights:
            if stem in stem_tfidf_dict.keys():
                tfidf_weight = stem_tfidf_dict[stem]
                if tfidf_weight > highest_word_tfidf:
                    highest_word_tfidf = tfidf_weight

        (freq, prox) = word_freq_prox[stem]
        #higher weight for words near the beginning or end of the document
        prox_mult = 10 #how important proximity is (lower: less important)
        prox_weight = (1 - prox)*prox_mult

        #higher weight for words that occur more frequently
        #NOTE: if this is higher, more words will have a higher weight
        #      associated with them
        freq_mult = 1.5#0.5 #how important frequency is (lower: less important)
        freq_weight = (float(freq)/max_freq)*freq_mult

        # if the word is in the intro or conclusion, use its frequency weight
        #    rather than calculating the frequency component
        #if stem in input_text.intro_conc_stems:
        #    freq_weight = freq

        #NOTE: I left the weight of this component at 0 because it might not be
        #      useful - but didn't know if there is absolutely no use for this.
        #higher weight for words with higher tfidf score, normalize tfidf score
        #between 0 and tfidf_mult
        tfidf_mult = 0#.5
        tfidf = (highest_word_tfidf / max_tfidf)*tfidf_mult

        #this power is useful for creating more discrete divisions between
        #   word ranks (i.e., the higher the power, the more "groups" of
        #   ranks
        power = len(input_text.filtered_tokens)
        if len(input_text.filtered_tokens) >= 10:
            power = 10

        word_weight_dict[stem] = (prox_weight*freq_weight+tfidf)**power
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
                    if word == actual_word and p not in pos[word]:
                        pos[word].append(p)
            #add the real word weight, the actual words (not the stems), and their
            #associated pos tags
            heapq.heappush(word_and_weights,(-word_weight, actual_words, \
                    [pos[actual_word] for actual_word in actual_words]))

    #print out up to 20 rows of weights, pos tags, and words
    for _ in xrange(max([20, len(weights.keys())])):
        curr_word = heapq.heappop(word_and_weights)
        logger.info("\tWeight, pos, word: %s %s \t%s" % (-curr_word[0], curr_word[1], curr_word[2]))

#Builds a list of possible titles out of the given input.
def build_titles(input_text):
    compatible_pos_templates = get_compatible_pos_templates(input_text)
    if not compatible_pos_templates:
        raise ValueError("No valid POS templates for input text.")

    titles = []
    for _ in xrange(NUM_TITLES):
        pos_template = random.choice(compatible_pos_templates)
        title = build_title_with_template(input_text, pos_template)

        titles.append(title)

    return titles

#Gets a list of POS templates that are compatible with the input text.
#A POS template can be rendered incompatible in the edge case that a
#given text does not have any words for the required POS in a template.
def get_compatible_pos_templates(input_text):
    compatible_templates = []
    for template in POS_TEMPLATES:
        valid = True
        for pos_element in template:
            if isinstance(pos_element, tuple):
                valid = np.any([is_available_pos(input_text, pos) for pos in pos_element])
            elif not is_manual_word(pos_element):
                valid = is_available_pos(input_text, pos_element)
            if not valid:
                break

        if valid:
            compatible_templates.append(template)

    return compatible_templates

#Checks to see if there is a POS template entry is a manually set word.
#This is done by looking for lowercase characters, which are not contained in POS tags.
def is_manual_word(word):
    return isinstance(word, str) and re.search('[^A-Z\$]', word)

#Checks if a part of speech is usable based on the input text.
#Usually whether the text contains any word of that part of speech,
#but adjectives and adverbs are always allowed since they are optional when building the title.
def is_available_pos(input_text, pos):
    return pos in input_text.used_pos or pos.startswith(ADJECTIVE) or pos.startswith(ADVERB)

#Builds a single title using a given POS template.
def build_title_with_template(input_text, pos_template):
    logger.info("Building title for POS template: %s" % pos_template)
    title_words = []
    #Keep track of used words to avoid duplicate non-stop-words in a title.
    title_word_set = set()
    title_stem_set = set()
    for pos_element in pos_template:
        if is_manual_word(pos_element):
            new_word = (pos_element, DEFAULT_TAG)
        else:
            is_multiple_pos = isinstance(pos_element, tuple)
            #Adjective and adverbs have a chance of being used, but are not guaranteed.
            if pos_element == ADJECTIVES and random.random() > ADJECTIVE_CHANCE:
                continue
            elif pos_element == ADVERBS and random.random() > ADVERB_CHANCE:
                continue

            if is_multiple_pos:
                #Add all POS lists if there are multiple options.
                possible_words = []
                for pos in pos_element:
                    possible_words += [(word, pos) for word in input_text.pos_tag_and_words[pos]]
            else:
                possible_words = [(word, pos_element) for word in input_text.pos_tag_and_words[pos_element]]

            if not possible_words:
                continue

            if pos_element in POS_USE_STOPWORDS:
                #Certain POS like articles and conjunctions are essentially all stopwords and are left unweighted.
                #Uniformly randomly choose between the ones in the document.
                new_word = random.choice(possible_words)
            else:
                #Use a weighted random of the words of the given POS according to their weights.
                weight_sum = 0
                possible_word_weights = []
                for possible_word in possible_words:
                    word, pos = possible_word
                    stem = input_text.filtered_bases_and_words[word]
                    if not stem:
                        stem = word
                    if stem in input_text.word_weights and word not in title_word_set and stem not in title_stem_set:
                        weight = input_text.word_weights[stem]
                        weight_sum += weight
                        possible_word_weights.append((possible_word, weight))

                if weight_sum == 0:
                    #All words for this POS are stopwords. Randomly choose between them.
                    new_word = random.choice(possible_words)
                else:
                    number_choice = random.uniform(0, weight_sum)
                    for word_tuple in possible_word_weights:
                        number_choice -= word_tuple[1]
                        if number_choice <= 0:
                            break

                    new_word = word_tuple[0]

        title_words.append(list(new_word))
        title_word_set.add(new_word[0])
        title_stem_set.add(input_text.filtered_bases_and_words[new_word[0]])

    title = form_title_from_words(input_text, title_words)

    return title

#Does postprocessing on a list of words to turn it into a proper title.
def form_title_from_words(input_text, title_words):
    for i, tagged_word in enumerate(title_words):
        word, pos = tagged_word

        if i < len(title_words) - 1 and word == 'a' or word == 'an':
            #Correct the use of 'a' and 'an' based on the next word's starting letter.
            next_word = title_words[i + 1][0]
            if next_word[0] in VOWELS:
                word = 'an'
            else:
                word = 'a'

        if i == 0 or word == 'I' or (pos not in POS_NOT_CAPITAL and word not in input_text.stop_words):
            #Capitalize title words except certain "less important" words.
            word = word.capitalize()

        tagged_word[0] = word

    title = ''
    for tagged_word in title_words:
        if title and tagged_word[0] not in PUNCTUATION:
            title += ' '
        title += tagged_word[0]

    return title

#Returns the title's score (i.e., summed weight of all the title's words)
def get_title_score(title, input_text):
    score = 0
    title = title.translate(None, string.punctuation)
    for word in title.split():
        word = word.lower()
        stem = input_text.filtered_bases_and_words[word]
        if stem!='': score += input_text.word_weights[stem]
    return score

#Return a list of titles and their scores, ordered from "best" (1) to
#   "worst" (0) relative to the sum of their composite word weights
def order_titles(titles, input_text):
    titles_ranked = []
    titles_temp_ranked = []
    titles_temp_ranked2 = []

    for title in titles:
        title_score = get_title_score(title, input_text)
        heapq.heappush(titles_temp_ranked, (-title_score,title))

    max_score = -1
    for i,_ in enumerate(titles):
        score, title = heapq.heappop(titles_temp_ranked) #scores are negative
        if i==0: max_score = score
        diff = max_score - score
        heapq.heappush(titles_temp_ranked2, (diff, title))
    for i,_ in enumerate(titles):
        diff, title = heapq.heappop(titles_temp_ranked2)
        if i==0: max_diff = diff
        heapq.heappush(titles_temp_ranked, ((diff/max_diff),title))
    for _ in titles:
        diff, title = heapq.heappop(titles_temp_ranked)
        titles_ranked.append((title, 1-diff))
    return titles_ranked

#Prints the list of generated titles to the user.
def print_titles(titles):
    logger.info("------ Print Generated Titles ------")
    for title in titles:
        #Don't use a logger here as this functionality should not be suppressed when changing the logger level.
        print title

if __name__ == "__main__":
    #Start the basic command line interface
    logger.info("------------ Starting ------------")

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", help="File name of .txt file with extension (e.g., text.txt)", type=str)
    args = parser.parse_args()

    main(args, use_rake=False, use_summa_text_rank=False, use_text_rank=False)
    logger.info("------------ Finished ------------")
