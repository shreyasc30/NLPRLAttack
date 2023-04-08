

from textattack.shared import WordEmbedding
from textattack.transformations import WordChange

import re


class WordChangeRLSwap(WordChange):
    """An abstract class that takes a sentence and transforms it by replacing several words with synonyms


    num_of_candidates (int): number of synonyms per swappable word
    """

    def __init__(self, num_candidates):
        self.num_candidates = num_candidates
        return


    def swapable_words(self, text_list):
        # removes stopwords and returns the remaining words
        stopwords = {"a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost",
                     "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another",
                     "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around",
                     "as",
                     "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides",
                     "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d",
                     "didn",
                     "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else",
                     "elsewhere",
                     "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first",
                     "for",
                     "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he",
                     "hence",
                     "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself",
                     "his",
                     "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's",
                     "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn",
                     "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my",
                     "myself",
                     "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none",
                     "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only",
                     "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per",
                     "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't",
                     "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the",
                     "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
                     "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout",
                     "thru",
                     "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve",
                     "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence",
                     "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever",
                     "whether",
                     "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within",
                     "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll",
                     "you're", "you've", "your", "yours", "yourself", "yourselves"}
        words = set(text_list) - stopwords
        return words

    def words2swap(self, text):
        # input = original text
        # output= original text, indices of swapable words in text

        text_list = list(text.split(" "))  # split text and convert to string
        wordz = self.swapable_words(text_list)  # get words that should be swapped

        mask_list = []
        for pattern in list(wordz):
            word_match = re.search(pattern, text)
            mask_list.append(word_match.span())
        return list(wordz), mask_list  # get positions of swapable words in the original text

    def recover_word_case(self, word, reference_word):
        """Makes the case of `word` like the case of `reference_word`.
        Supports lowercase, UPPERCASE, and Capitalized.
        """
        if reference_word.islower():
            return word.lower()
        elif reference_word.isupper() and len(reference_word) > 1:
            return word.upper()
        elif reference_word[0].isupper() and reference_word[1:].islower():
            return word.capitalize()
        else:
            # if other, just do not alter the word's case
            return word

    def fetch_replacement_words(self, word, num_of_candidates, embedding):
        try:
            word_id = embedding.word2index(word.lower())
            nnids = embedding.nearest_neighbours(word_id, num_of_candidates)
            candidate_words = []
            for i, nbr_id in enumerate(nnids):
                nbr_word = embedding.index2word(nbr_id)
                candidate_words.append(self.recover_word_case(nbr_word, word))
            return candidate_words
        except KeyError:
            # This word is not in our word embedding database, so return an empty list.
            return []

    def change_words(self, text, num_of_candidates=2):
        text_list, mask_list = self.words2swap(text)
        embedding = WordEmbedding.counterfitted_GLOVE_embedding()
        word_dict = {}
        for word in text_list:
            candidate_word_list = self.fetch_replacement_words(word, num_of_candidates, embedding)
            word_dict[word] = candidate_word_list

        return word_dict

    def remove_null_swaps(self, text_dict):
        t_dict = {}
        for key, value in list(text_dict.items()):
            if value != []:
                t_dict[key] = value
        return t_dict

    def compress_changed_words(self, t_dict, text, num_of_candidates=2):
        list_sentences_with_changes = []
        for num in range(num_of_candidates):
            text_ = text
            for key in t_dict.keys():
                re = str(' ' + key + ' ')
                subb = str(' ' + t_dict[key][num] + ' ')
                change = re.sub(re, subb, text_)
                text_ = change
            list_sentences_with_changes.append(change)
        return list_sentences_with_changes

