from textattack.transformations import Transformation
import re


class WordChange(Transformation):
    """An abstract class that takes a sentence and transforms it by replacing several words with synonyms


    num_of_candidates (int): number of synonyms per swappable word
    """

    def __init__(self, num_candidates):
        self.num_candidates = num_candidates
        return

    def _get_transformations(self, text):
        word_dict = self.change_words(text, num_candidates=self.num_candidates)
        word_dict = self.remove_null_swaps(word_dict)
        list_sentences_with_changes = self.compress_changed_words(word_dict, text, num_of_candidates)
        return list_sentences_with_changes

