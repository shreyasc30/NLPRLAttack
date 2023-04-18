import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from textattack.models.RL_method import SimpleRLLSTM
torch.manual_seed(1)


class RLWrapper(nn.Module):
    def __init__(self, embedding_dim, hidden_dim_lstm, num_hidden_layers_lstm, lstm_out_size, output_size, max_length_sentence, max_swappable_words) -> None:
        super(RLWrapper, self).__init__()
        # NOTE: skip the embedding here since we're already passing in gloVe embeddings

        self.sentence_lstm = SimpleRLLSTM(embedding_dim, hidden_dim_lstm, lstm_out_size, num_hidden_layers_lstm, max_length_sentence)
        self.word_candidates_lstm = SimpleRLLSTM(embedding_dim, hidden_dim_lstm, lstm_out_size, num_hidden_layers_lstm, max_swappable_words)
        
        # self.fc_indicator_to_embedding = nn.Linear(max_length_sentence, lstm_out_size)

        self.fc_words_and_indicators = nn.Linear(lstm_out_size + max_length_sentence, lstm_out_size * 2)

        self.fc_sentence = nn.Linear(lstm_out_size + max_length_sentence, lstm_out_size)

        self.fc_final0 = nn.Linear(lstm_out_size * 3, lstm_out_size)
        self.fc_final1 = nn.Linear(lstm_out_size, output_size)

    def forward(self, sentence_embeddings, word_embeddings, indicators):

        sentence_out = self.sentence_lstm(sentence_embeddings)
        word_candidate_out = self.word_candidates_lstm(word_embeddings)

        # ind_embedding = F.relu(self.fc_indicator_to_embedding(indicators))

        word_and_indicators = F.relu(self.fc_words_and_indicators(torch.concat([word_candidate_out, indicators], dim=1)))
        sentence_embedding = F.relu(self.fc_sentence(torch.concat([sentence_out, indicators], dim=1)))

        out = F.relu(self.fc_final0(torch.concat([sentence_embedding, word_and_indicators], dim=1)))
        out = self.fc_final1(out)

        return out