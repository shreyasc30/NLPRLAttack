from textattack.transformations import Transformation
from textattack.shared import AttackedText
import re

def process_text(text):
    
    # new_text_list = list(text.replace('\\', ' ').replace("'",' ').replace("''", ' ').replace("!@#$%^&*()[]{};:',./<>?|`~-=_+", ' ').split(" ")) #
    new_text_list = list(text.replace('\\', ' ').replace('[','').replace(']',' ').replace('(',' ').replace(')',' ').split(" "))
    processed_text = ' '.join(new_text_list)
    return processed_text



class WordChange(Transformation):
    """An abstract class that takes a sentence and transforms it by replacing several words with synonyms
    num_of_candidates (int): number of synonyms per swappable word
    """

    def __init__(self, num_candidates, dissimilar_swaps = False):  #
        self.num_candidates = num_candidates
        self.dissimilar_swaps = dissimilar_swaps  # True or False  
        return

    def _get_transformations(self, text, indices_to_modify):
        text = text.text  # AttackText object has attribute text to access the full string
        text = process_text(text)
        if self.dissimilar_swaps == True: #
            word_dict = self.change_words(text, num_candidates=3)
            word_dict = self.remove_null_swaps(word_dict)
            word_dict = self.get_best_swap(word_dict, num_candidates=3)
        else:
            word_dict = self.change_words(text, num_candidates=self.num_candidates)
            word_dict = self.remove_null_swaps(word_dict)
        list_sentences_with_changes = self.compress_changed_words(word_dict, text, self.num_candidates)
        attacked_text_list = [AttackedText(l) for l in list_sentences_with_changes]
        return attacked_text_list
