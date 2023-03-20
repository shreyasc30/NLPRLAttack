# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder

from textattack.transformations import WordSwapEmbedding


from textattack.augmentation import Augmenter


def text_to_transformed_text(original_text, num_of_transformations=10):

    stopwords = {"a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost",
                 "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another",
                 "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as",
                 "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides",
                 "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn",
                 "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere",
                 "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for",
                 "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence",
                 "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
                 "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's",
                 "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn",
                 "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself",
                 "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none",
                 "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only",
                 "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per",
                 "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't",
                 "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the",
                 "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
                 "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru",
                 "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve",
                 "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence",
                 "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether",
                 "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within",
                 "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll",
                 "you're", "you've", "your", "yours", "yourself", "yourselves"}

    transformation = WordSwapEmbedding(max_candidates=5)

    constraints = [RepeatModification(), StopwordModification(stopwords=stopwords),
                   WordEmbeddingDistance(min_cos_sim=0.5), PartOfSpeech(allow_verb_noun_swap=True)]
    use_constraint = UniversalSentenceEncoder(
        threshold=0.840845057,
        metric="angular",
        compare_against_original=False,
        window_size=15,
        skip_text_shorter_than_window=True,
    )
    constraints.append(use_constraint)

    augmenter = Augmenter(transformation=transformation, constraints=constraints, pct_words_to_swap=0.5,
                          transformations_per_example=num_of_transformations)

    return augmenter.augment(original_text)

