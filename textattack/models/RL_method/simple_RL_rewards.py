import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
# tokenizer = AutoTokenizer.from_pretrained("2023-03-18-09-14-37-047051/best_model")
# model = AutoModelForTokenClassification.from_pretrained("2023-03-18-09-14-37-047051/best_model")
# TODO: replace the above constants with input from our model -- kept here as an example, but there's a way to call this model locally somehow

class RewardSelector():

    def __init__(self, path) -> None:
        self.path = path
        self.tokenizer = None
        self.model = None
    
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForTokenClassification(self.path)
    
    def tokenize(self, sequence):
        return self.tokenizer(sequence)

    # input text is the original text
    # replacements is the list of list of possible replacements
    # mask is 0 (ignore) or 1 (replace word)
    def forward(self, input_text, replacements, mask):
        possible_inputs = []
        for w_idx in range(len(input_text)):
            if mask[w_idx]:
                cpy = input_text.copy()
                for repnum in range(len(replacements[w_idx])):
                    cpy[w_idx] = replacements[repnum]
                    possible_inputs.append(cpy)
        
        # now that we have all of our possible inputs, we want to
        # tokenize all of these and pass through the model, returning logits so we can see which one drives them most away from correct class
        # ^ that implies some sort of greedy reward formulation, is this true?

        # we can replace most of this if the model is just callable somehow through textattack, all we need is the logic above -- but even that can be extracted since we need the same logic for the MLP forward pass
