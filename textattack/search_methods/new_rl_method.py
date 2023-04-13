import numpy as np
import torch
import torch.nn.functional as F
import collections

import regex as re
import copy

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared import AttackedText

from textattack.models.helpers import GloveEmbeddingLayer
from textattack.shared import WordEmbedding
from textattack.models.RL_method import RLWrapper

Transition = collections.namedtuple(
    "Transition",
    "state word_candidates action reward next_state next_word_candidates done legal_actions_mask indicators")


class RLWordSwap(SearchMethod): 
    def __init__(self, lr, gamma, batch_size):
        # TODO: Figure out the mebedding space
        self.buffer = []

        self.max_length_rollout = 10
        self.batch_size = batch_size
        self.gamma = gamma 

        # Hyperparameters for reward function 
        self.lambda_logit_diff = 1
        self.lambda_search_success = 5
        self.constant_reward = -.1

        # Cache for target model reward scores 
        self.cache = {}  # maps sentence to score 
        self.max_cache_size = int(1e6)

        # TODO: Calculate input size 
        # Output size = action space 
        # self.word2idx = WordEmbedding.counterfitted_GLOVE_embedding()._word2index
        self.embedding = GloveEmbeddingLayer(emb_layer_trainable=False) # GloveTokenizer(word_id_map=self.word2idx, pad_token_id=len(self.word2idx), unk_token_id=len(self.word2idx)+1, max_length=256)

        self.embedding_size = 200 
        self.max_num_words_in_sentence = 80  # These two values are dummy variables for now
        self.max_num_words_swappable_in_sentence = 30

        # For MLP, the input size has 3 parts: sentence embedding, word swap embedding, and the indicators for swappable token indices
        # self.input_size = self.embedding_size * (self.max_num_words_in_sentence + self.max_num_words_swappable_in_sentence) + self.max_num_words_swappable_in_sentence  
        self.num_actions = self.max_num_words_swappable_in_sentence + 1 #  +1 for stop action
        self.model = RLWrapper(embedding_dim=self.embedding_size,
                                hidden_dim_lstm=64,
                                num_hidden_layers_lstm=1,
                                lstm_out_size=128,
                                output_size=self.num_actions,
                                max_length_sentence=self.max_num_words_in_sentence,
                                max_swappable_words=self.max_num_words_swappable_in_sentence)# SimpleRLLSTM(self.input_size, self.num_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.max_sentence_length = 0
        return 

    def take_out_extra_spaces(self, sentence):
        return re.sub(' +', ' ', sentence)
    
    def perform_search(self, initial_result):
        # Initial result is the input 
        self._search_over = False 
        original_text = initial_result.attacked_text
        curr_state = initial_result

        original_text = self.take_out_extra_spaces(original_text.text) 
        edited_attacked_text = AttackedText(original_text)

        self.max_sentence_length = max(self.max_sentence_length, len(original_text.split(' ')))
        # return curr_state
        
        transformed_texts = self.get_transformations(edited_attacked_text, None)
        # Get the indices of the candidate words to switch out 

        tokens = original_text.split(' ')
        indicators = [0 for _ in range(len(tokens))]
        original_words = {}  # maps the index of the token in the sentence to the original token
        transformed_words = {}  # maps the index of the token in the sentence to a list of token options (including the original token)
        for attack_text_object in transformed_texts:
            curr_text = attack_text_object.text
            transformed_tokens = curr_text.split(' ')
            assert len(tokens) == len(transformed_tokens)
            for j in range(len(tokens)):
                # If the original token is different from the transformed token
                if tokens[j] != transformed_tokens[j]:
                    # Non-masked at this token because can be modified
                    indicators[j] = 1

                    # If the original token hasn't been accounted for, then add it to the original word list
                    if j not in original_words.keys(): 
                        original_words[j] = tokens[j]

                    # If a new list of word modification hasn't been created for this index 
                    if j not in transformed_words.keys():
                        transformed_words[j] = [transformed_tokens[j]]
                    else:
                        # If there is an existing list, append the modified word if not in there already
                        curr_options = transformed_words[j]
                        if transformed_tokens[j] not in curr_options:
                            curr_options.append(transformed_tokens[j])
        # Pad the indicators
        num_pad_indicator = max(self.max_num_words_in_sentence - len(indicators), 0)
        indicators = indicators + ([0] * num_pad_indicator)

        # Create legal actions mask for this example
        legal_actions_mask = [1 if i < sum(indicators) else 0 for i in range(self.max_num_words_swappable_in_sentence)] + [1]

        action_to_index = {}  # maps the action index to a tuple (index in sentence, index of the word options for that index)
        curr_action = 0
        for index_in_sentence in original_words.keys():  # i is index in the sentence
            num_options = transformed_words[index_in_sentence]
            for index_in_word_options in range(len(num_options)):  # j is the index of the word options for a particular index in the sentence
                action_to_index[curr_action] = (index_in_sentence, index_in_word_options)
                curr_action += 1
        
        stop_action = self.num_actions - 1

        # Create the word candidates that will jump start the RL learning process
        word_candidates = {}
        for index_in_sentence, word_list in transformed_words.items():
            word_list.sort(reverse=False) # make these alphabetical order for consistency in state space
            word_candidates[index_in_sentence] = word_list  

        curr_state_string = self.take_out_extra_spaces(curr_state.attacked_text.text) 
        curr_state = AttackedText(curr_state_string)

        curr_state_tokens = curr_state.text.split(' ')
        sentence_embedding, word_embedding, indicator_embedding = self.create_embeddings(curr_state_tokens, word_candidates, indicators)
        prev_word_candidates = copy.deepcopy(word_candidates)
 
        for i in range(self.max_length_rollout):

            # curr_state_tokens = curr_state.text.split(' ') 

            # action, probs, state_embedding = self.get_action(curr_state_tokens, word_candidates, indicators, legal_actions_mask)

            action, probs = self.get_action(sentence_embedding, word_embedding, indicator_embedding, legal_actions_mask)

            if action != stop_action:  # if we swap out a word
                index_in_sentence, index_of_word_list = action_to_index[action]

                word_to_replace_with = word_candidates[index_in_sentence][index_of_word_list]

                # Modify word candidates by reinserting the previous word and taking out the word replacement
                word_candidates[index_in_sentence].remove(word_to_replace_with)
                word_candidates[index_in_sentence].append(curr_state_tokens[index_in_sentence])
                word_candidates[index_in_sentence].sort(reverse=False)

                next_state = curr_state.replace_word_at_index(index_in_sentence, word_to_replace_with)

            r = self.reward_function(curr_state, action, next_state, stop_action)

            next_state_tokens = next_state.text.split(' ')
            next_sentence_embedding, next_word_embedding, next_indicator_embedding = self.create_embeddings(next_state_tokens, word_candidates, indicators)

            # TODO: Fix so that we store sentences
            self.buffer.append(Transition(state=curr_state_tokens, word_candidates=prev_word_candidates, 
                                          action=action, reward=r, next_state=next_state_tokens, 
                                          next_word_candidates=word_candidates, done=self._search_over, 
                                          legal_actions_mask=legal_actions_mask, indicators=indicators))  # legal_actions_mask is same for each rollout

            # Preparing all variables for next iteration
            curr_state = next_state
            curr_state_tokens = curr_state.text.split(' ') 
            prev_word_candidates = copy.deepcopy(word_candidates)

            sentence_embedding = next_sentence_embedding
            word_embedding = next_word_embedding
            indicator_embedding = next_indicator_embedding


            if action == stop_action:
                break
        print("Buffer: ", len(self.buffer))
        if len(self.buffer) > self.batch_size:
            self.update()

        curr_state = self.cache[curr_state.text][0]

        return curr_state
        

    def get_action(self, sentence_embedding, word_embedding, indicator_embedding, legal_actions_mask):
        probs = self.get_action_probs(sentence_embedding, word_embedding, indicator_embedding, legal_actions_mask)

        # Remember to insert the additional action at the very END corresponding to stop action
        action = probs.multinomial(num_samples=1, replacement=False)[0].item()

        return action, probs

    def get_action_probs(self, sentence_embedding, word_embedding, indicator_embedding, legal_actions_mask):
        if sentence_embedding.dim() == 2:  # if a single datapoint 
            sentence_embedding = torch.unsqueeze(sentence_embedding, 0)
            word_embedding = torch.unsqueeze(word_embedding, 0)
            indicator_embedding = torch.unsqueeze(indicator_embedding, 0)
            legal_actions_mask = [legal_actions_mask]

        output = self.model(torch.FloatTensor(sentence_embedding), torch.FloatTensor(word_embedding), indicator_embedding)

        probs = F.softmax(output, dim=1)


        probs = probs * torch.FloatTensor(legal_actions_mask)


        renorm = torch.sum(probs, dim=1)
        renorm = torch.reshape(renorm, (-1, 1))

        probs = probs / renorm

        return probs



    def query_word_to_id(self, word):
        try:
            return self.embedding.word2id[word.lower()]
        except KeyError:
            return self.embedding.oovid  # out of vocabulary id
    

    
    def create_embeddings(self, sentence_tokens, word_candidates, indicators):
        word_candidates_sorted = [list(map(lambda x: x.lower(), word_candidates[index])) for index in sorted(word_candidates.keys())]
        word_candidates_sorted = [lst[0] for lst in word_candidates_sorted]  # Assume that we only have 1 candidate to swap
        # TODO: Reconvert casing

        sentence_embedding_ids = [self.query_word_to_id(w) for w in sentence_tokens]  
        num_pad_tokens = max(self.max_num_words_in_sentence - len(sentence_embedding_ids), 0)
        sentence_embedding_ids = sentence_embedding_ids + ([self.embedding.padid] * num_pad_tokens)

        sentence_embedding = self.embedding(torch.tensor(sentence_embedding_ids, dtype=torch.int32))

        assert not any([len(w) == 0 for w in word_candidates_sorted])
        word_candidate_ids = [self.query_word_to_id(w) for w in word_candidates_sorted]
        num_pad_tokens = max(self.max_num_words_swappable_in_sentence - len(word_candidate_ids), 0)
        word_candidate_ids = word_candidate_ids + ([self.embedding.padid] * num_pad_tokens)

        word_candidate_embedding = self.embedding(torch.tensor(word_candidate_ids, dtype=torch.int32))

        # result = torch.concat([torch.flatten(sentence_embedding), torch.flatten(word_candidate_embedding), torch.Tensor(indicators)])

        return sentence_embedding, word_candidate_embedding, torch.Tensor(indicators)

    
    def get_goal_result_wrapper(self, state):
        # State is an attacked_text object
        if state.text in self.cache:
            goal_result = self.cache[state.text]
            self.cache.pop(state.text)  # remove and reinsert to push it to end of ordered dictionary
        else:
            goal_result, _ = self.get_goal_results([state])
            s_score = goal_result[0].score 
        self.cache[state.text] = goal_result  # insert to end of ordered dictionary 

        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)  # remove the oldest entry in dictionary 
        return goal_result

    def reward_function(self, s, a, next_s, stop_action):
        # reward an increase in the target output class on the target model 
        s_goal_result = self.get_goal_result_wrapper(s)
        next_s_goal_result = self.get_goal_result_wrapper(next_s)

        s_score = s_goal_result[0].score 
        next_s_score = next_s_goal_result[0].score
        
        reward_logit_diff = next_s_score - s_score
        reward_success = self.lambda_search_success if self._search_over else 0

        if a == stop_action and not self._search_over:
            return -self.lambda_search_success + self.constant_reward
    
        return self.lambda_logit_diff * reward_logit_diff + reward_success + self.constant_reward
    
    @property
    def is_black_box(self):
        return False

    def update(self):
        # Given sequence of transitions, work backwards to calculate rewards to go for all trajectories 
        # As long as we properly account for done, we should be able to do it in one swoop 

        sentence_embedding = []
        word_embedding = []
        indicator_embedding = []

        next_sentence_embedding = []
        next_word_embedding = []
        next_indicator_embedding = []

        for t in self.buffer:
            curr_sentence_embedding, curr_word_embedding, curr_indicator_embedding = self.create_embeddings(t.state, t.word_candidates, t.indicators)
            sentence_embedding.append(curr_sentence_embedding)
            word_embedding.append(curr_word_embedding)
            indicator_embedding.append(curr_indicator_embedding)
            
            curr_sentence_embedding, curr_word_embedding, curr_indicator_embedding = self.create_embeddings(t.next_state, t.next_word_candidates, t.indicators)
            next_sentence_embedding.append(curr_sentence_embedding)
            next_word_embedding.append(curr_word_embedding)
            next_indicator_embedding.append(curr_indicator_embedding)
            
        actions = [t.action for t in self.buffer]
        rewards = [t.reward for t in self.buffer]
        dones = [t.done for t in self.buffer]
        legal_actions_masks = [t.legal_actions_mask for t in self.buffer]

        rtg = [0 for _ in range(len(self.buffer) + 1)]
        b = np.mean(rewards)
        for i in range(len(self.buffer) - 1, -1, -1):
            rtg[i] = (rewards[i] - b) + self.gamma * rtg[i+1] * (1 - dones[i])
        rtg = rtg[:-1]

        action_probs = self.get_action_probs(torch.stack(sentence_embedding), torch.stack(word_embedding), torch.stack(indicator_embedding), torch.FloatTensor(legal_actions_masks))

        selected_action_probs = torch.gather(action_probs, 1, torch.reshape(torch.tensor(actions), (-1, 1)))

        # Use actions to query gather from action_probs
        loss = -torch.mean(torch.log(selected_action_probs) * torch.tensor(rtg, dtype=torch.float32))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print("Iteration loss: {}".format(loss.item()))

        self.buffer = []
        return 