"""
BERT-Attack:
============================================================

(BERT-Attack: Adversarial Attack Against BERT Using BERT)

.. warning::
    This attack is super slow
    (see https://github.com/QData/TextAttack/issues/586)
    Consider using smaller values for "max_candidates".

"""
from textattack import Attack
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapMaskedLM

from textattack.transformations import WordChangeRLSwap
from textattack.search_methods import RLWordSwap

from .attack_recipe import AttackRecipe


class NLPRLAttack(AttackRecipe):
    """Li, L.., Ma, R., Guo, Q., Xiangyang, X., Xipeng, Q. (2020).

    BERT-ATTACK: Adversarial Attack Against BERT Using BERT

    https://arxiv.org/abs/2004.09984

    This is "attack mode" 1 from the paper, BAE-R, word replacement.
    """

    @staticmethod
    def build(model_wrapper):
        # [from correspondence with the author]
        # Candidate size K is set to 48 for all data-sets.
        transformation = WordChangeRLSwap(num_candidates=1)# WordSwapMaskedLM(method="bert-attack", max_candidates=48)
        #
        # Don't modify the same word twice or stopwords.
        #
        constraints = []
        # Goal is untargeted classification.
        #
        goal_function = UntargetedClassification(model_wrapper)
        #
        # "We first select the words in the sequence which have a high significance
        # influence on the final output logit. Let S = [w0, ··· , wi ··· ] denote
        # the input sentence, and oy(S) denote the logit output by the target model
        # for correct label y, the importance score Iwi is defined as
        # Iwi = oy(S) − oy(S\wi), where S\wi = [w0, ··· , wi−1, [MASK], wi+1, ···]
        # is the sentence after replacing wi with [MASK]. Then we rank all the words
        # according to the ranking score Iwi in descending order to create word list
        # L."
        search_method = RLWordSwap(lr=3e-4, gamma=.99, batch_size=4)

        return Attack(goal_function, constraints, transformation, search_method)
