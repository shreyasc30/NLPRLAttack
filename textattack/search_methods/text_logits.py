from transformers import RobertaTokenizer, RobertaModel


class text_logits:
  def __init__(self, model_path, text):
    self.model_path = model_path
    self.text = text

  def get_hidden_state_logits(self):
  	# reutrns the outputs from last 4 hiddden states of the model after inference on text

  	tokenizer = RobertaTokenizer.from_pretrained(self.model_path)
  	encoded_input = tokenizer(self.text, return_tensors='pt')
  	model = RobertaModel.from_pretrained(self.model_path, output_hidden_states=True)
  	output = model(**encoded_input)

  	hidden_states = output[2]
  	last_four_hidden = hidden_states[-4:]

  	return last_four_hidden

