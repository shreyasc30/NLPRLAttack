import numpy as np
import torch
from torch.nn.functional import softmax
import collections

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.models.tokenizers import GloveTokenizer
from textattack.shared import WordEmbedding
from textattack.models.RL_method import SimpleRLMLP

Transition = collections.namedtuple(
    "Transition",
    "state action reward next_state done legal_actions_mask")


class RLWordSwap(SearchMethod): 
    def __init__(self, lr, gamma, batch_size):
        # TODO: Figure out the mebedding space
        self.buffer = []

        self.max_length_rollout = 50
        self.batch_size = batch_size

        # Hyperparameters for reward function 
        self.lambda_logit_diff = 1
        self.lambda_search_success = 5
        self.constant_reward = -.1

        # Cache for target model reward scores 
        self.cache = {}  # maps sentence to score 
        self.max_cache_size = int(1e9)

        # TODO: Calculate input size 
        # Output size = action space 
        self.word2idx = WordEmbedding.counterfitted_GLOVE_embedding()._word2index
        self.tokenizer = GloveTokenizer(word_id_map=self.word2idx, pad_token_id=len(self.word2idx), unk_token_id=len(self.word2idx)+1, max_length=256)

        self.input_size = 256
        self.output_size = 100 + 1 # This is a dummy variable for now. +1 for stop action
        self.model = SimpleRLMLP(self.input_size, self.output_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        return 
    
    def perform_search(self, initial_result):
        # Initial result is the input 
        self._search_over = False 
        original_text = initial_result.attacked_text
        curr_state = initial_result.attacked_text

        transformed_texts = self.get_transformations(curr_state, None)
        # Get the indices of the candidate words to switch out 
        original_text = original_text.text

        tokens = original_text.split(' ')
        indicators = [0 for _ in range(len(tokens))]
        original_words = {}  # maps the index of the token in the sentence to the original token
        transformed_words = {}  # maps the index of the token in the sentence to a list of token options (including the original token)
        for attack_text_object in transformed_texts:
            curr_text = attack_text_object.text
            transformed_tokens = curr_text.split(' ')
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

        action_to_index = {}  # maps the action index to a tuple (index in sentence, index of the word options for that index)
        curr_action = 0
        for index_in_sentence in original_words.keys():  # i is index in the sentence
            num_options = transformed_words[index_in_sentence]
            for index_in_word_options in range(len(num_options)):  # j is the index of the word options for a particular index in the sentence
                action_to_index[curr_action] = (index_in_sentence, index_in_word_options)
                curr_action += 1
        
        stop_action = len(action_to_index) # the last action will always correspond to stopping 

        # Create the word candidates that will jump start the RL learning process
        word_candidates = {}
        for index_in_sentence, word_list in transformed_words.items():
            word_list.sort(reverse=False) # make these alphabetical order for consistency in state space
            word_candidates[index_in_sentence] = word_list  

        curr_state = curr_state.text
        curr_state = curr_state.split(' ')  
        for i in range(self.max_length_rollout):
            
            action, probs = self.get_action(curr_state, word_candidates, indicators)

            next_state = curr_state.copy()

            if action != stop_action:  # if we swap out a word
                index_in_sentence, index_of_word_list = action_to_index[action]

                next_state[index_in_sentence] = word_candidates[index_in_sentence][index_of_word_list]

                # Modify word candidates by reinserting the previous word and taking out the word replacement
                word_candidates[index_in_sentence].remove(curr_state[index_in_sentence])
                word_candidates[index_in_sentence].append(next_state[index_in_sentence])
                word_candidates[index_in_sentence].sort(reverse=False)

            r = self.reward_function(curr_state, action, next_state, stop_action)

            self.buffer.append(Transition(state=curr_state, action=action, reward=r, next_state=next_state, done=self._search_over))

            curr_state = new_state

        if len(self.buffer) > self.batch_size:
            self.update()

    def get_action(self, curr_attacked_text, word_candidates, indicators):

        embedding = self.create_embeddings(curr_attacked_text, word_candidates, indicators)
        output = self.model(torch.FloatTensor(embedding))

        probs = torch.softmax(output)

        action = torch.sample(probs, dim=1)

        return action, probs
    
    def create_embeddings(self, sentence_tokens, word_candidates, indicators):
        word_candidates_sorted = [list(map(lambda x: x.lower(), word_candidates[index])) for index in sorted(word_candidates.keys())]
        word_candidates_sorted = [lst[0] for lst in word_candidates_sorted]
        # TODO: Reconvert casing
        """
        sentence_embedding = []
        for word in sentence_tokens:
            if len(word) > 0:
                try:
                    sentence_embedding.append(self.tokenizer[self.tokenizer.word2index(word.lower())])
                except KeyError:
                    word = ''
        """ 
        sentence_tokens = [w.lower() for w in sentence_tokens if len(w) > 0]
        for w in sentence_tokens:
            if len(w) > 0:
                try:
                    print('manual encoding: ', self.word2idx[w])
                    # print('tokenizer ', self.tokenizer.encode(w))
                except KeyError:
                    pass
        # print('manual encoding: ', [self.word2idx[w] for w in sentence_tokens if len(w) > 0])
        sentence_embedding = self.tokenizer.encode(' '.join(sentence_tokens))
        # sentence_embedding = self.tokenizer.batch_encode([sentence_tokens])

        # TODO: Insert padding and reconvert casing
        word_candidate_embedding = self.tokenizer.encode(' '.join(word_candidates_sorted))

        return sentence_embedding + word_candidate_embedding + indicators

    
    def get_goal_score(self, state):
        if state in self.cache:
            s_score = self.cache[state]
            self.cache.pop(state)  # remove and reinsert to push it to end of ordered dictionary
        else:
            original_results, _ = self.get_goal_results([state])
            s_score = original_results[0].score 
        self.cache[state] = s_score  # insert to end of ordered dictionary 

        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)  # remove the oldest entry in dictionary 
        return s_score

    def reward_function(self, s, a, next_s, stop_action):
        # reward an increase in the target output class on the target model 
        s_score = self.get_goal_score(s)
        next_s_score = self.get_goal_score(next_state)
        
        reward_logit_diff = next_s_score - s_score
        reward_success = self.lambda_search_success if self._search_over else 0

        if a == stop_action and not self._search_over:
            return -self.lambda_search_success + self.constant_reward
    
        return self.lambda_logit_diff * reward_logit_diff + reward_success + self.constant_reward
    
    def is_black_box(self):
        return False

    def update(self):
        # Given sequence of transitions, work backwards to calculate rewards to go for all trajectories 
        # As long as we properly account for done, we should be able to do it in one swoop 

        states = [t.state for t in self.buffer]
        actions = [t.action for t in self.buffer]
        rewards = [t.reward for t in self.buffer]
        next_state = [t.next_state for t in self.buffer]
        dones = [t.done for t in self.buffer]
        legal_actions_masks = [t.legal_actions_mask for t in self.buffer]

        rtg = [0 for _ in range(len(self.buffer) + 1)]
        b = np.mean(rewards)
        for i in range(len(self.buffer) - 1, -1, -1):
            rtg[i] = (rewards[i] - b) + self.gamma * rtg[i+1] * (1 - dones[i])
        rtg = rtg[:-1]

        _, action_probs = self.get_action_probs(states)
        loss = -torch.mean(torch.log(action_probs) * torch.tensor(rtg, dtype=torch.float32))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.buffer = []
        return 