from transformers import RobertaTokenizer, RobertaModel
import torch


class text_logits:
	def __init__(self, model_path):
		self.model_path = model_path
		self.model = RobertaModel.from_pretrained(self.model_path, output_hidden_states=True)
		self.tokenizer = RobertaTokenizer.from_pretrained(self.model_path)
    
	def get_hidden_state_logits(self, text, mode='average'):
  	# reutrns the outputs from last 4 hiddden states of the model after inference on text
		encoded_input = self.tokenizer(text, return_tensors='pt')
		output = self.model(**encoded_input)
		
		hidden_states = output[2]
		last_four_hidden = hidden_states[-4:]
		l4h = None

		if mode == 'sum':
			l4h = torch.stack(last_four_hidden).sum(0)
			return l4h
		
		elif mode == 'average':
			l4h = torch.stack(last_four_hidden).mean(0)
			return l4h

		elif mode == 'max':
			l4h = torch.stack(last_four_hidden).max(0)
			return l4h
		else:
			raise Exception("invalid concat mode specified for logits")