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

from textattack.shared import utils

from textattack.transformations.word_changes import process_text

Transition = collections.namedtuple(
    "Transition",
    "state word_candidates action reward next_state next_word_candidates done legal_actions_mask indicators")


class RLWordSwap(SearchMethod): 
    def __init__(self, lr, gamma, batch_size):
        # TODO: Figure out the mebedding space
        self.buffer = []
        self.batch_size = batch_size
        self.gamma = gamma

        # DQN Parameters
        self.max_length_rollout = 250
        self.min_steps_warm_up = 200000
        self.update_target_every = 5000
        self.max_buffer_size = 1e6
        self.updates_so_far = 0

        # Exploration Parameters
        self.epsilon_init = 1
        self.epsilon_final = .1
        self.epsilon = self.epsilon_init
        self.epsilon_num_steps = 200000

        # Hyperparameters for reward function 
        self.lambda_logit_diff = 100
        self.lambda_search_success = 10
        self.constant_reward = -.02

        # Cache for target model reward scores 
        self.cache = {}  # maps sentence to score 
        self.max_cache_size = int(1e6)

        # TODO: Calculate input size 
        # Output size = action space 
        # self.word2idx = WordEmbedding.counterfitted_GLOVE_embedding()._word2index
        self.embedding = GloveEmbeddingLayer(emb_layer_trainable=False) # GloveTokenizer(word_id_map=self.word2idx, pad_token_id=len(self.word2idx), unk_token_id=len(self.word2idx)+1, max_length=256)

        self.embedding_size = 200 
        self.max_num_words_in_sentence = 137 # 137  # These two values are manually inputed values based on the dataset
        self.max_num_words_swappable_in_sentence = 61 # 61

        # For MLP, the input size has 3 parts: sentence embedding, word swap embedding, and the indicators for swappable token indices
        # self.input_size = self.embedding_size * (self.max_num_words_in_sentence + self.max_num_words_swappable_in_sentence) + self.max_num_words_swappable_in_sentence  
        self.num_actions = self.max_num_words_swappable_in_sentence + 1 #  +1 for stop action
        self.model = RLWrapper(embedding_dim=self.embedding_size,
                                hidden_dim_lstm=128,
                                num_hidden_layers_lstm=1,
                                lstm_out_size=200,
                                output_size=self.num_actions,
                                max_length_sentence=self.max_num_words_in_sentence,
                                max_swappable_words=self.max_num_words_swappable_in_sentence)# SimpleRLLSTM(self.input_size, self.num_actions)
        
        self.target_model = RLWrapper(embedding_dim=self.embedding_size,
                                hidden_dim_lstm=128,
                                num_hidden_layers_lstm=1,
                                lstm_out_size=200,
                                output_size=self.num_actions,
                                max_length_sentence=self.max_num_words_in_sentence,
                                max_swappable_words=self.max_num_words_swappable_in_sentence)
        
        self.model.to(utils.device)
        self.target_model.to(utils.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.episode_returns = []
        return 

    def take_out_extra_spaces(self, sentence):
        return re.sub(' +', ' ', sentence)

    # Overwriting parent method __call__ to support getting is_evaluation
    def __call__(self, initial_result, is_evaluation=True):
        """Ensures access to necessary functions, then calls
        ``perform_search``"""
        if not hasattr(self, "get_transformations"):
            raise AttributeError(
                "Search Method must have access to get_transformations method"
            )
        if not hasattr(self, "get_goal_results"):
            raise AttributeError(
                "Search Method must have access to get_goal_results method"
            )
        if not hasattr(self, "filter_transformations"):
            raise AttributeError(
                "Search Method must have access to filter_transformations method"
            )

        result = self.perform_search(initial_result, is_evaluation)
        # ensure that the number of queries for this GoalFunctionResult is up-to-date
        result.num_queries = self.goal_function.num_queries
        return result
    
    def perform_search(self, initial_result, is_evaluation=True, constrain=0):
        # Initial result is the input 
        self._search_over = False 
        original_text = initial_result.attacked_text
        curr_state = initial_result

        original_text = self.take_out_extra_spaces(original_text.text) 
        edited_attacked_text = AttackedText(original_text)
        
        transformed_texts = self.get_transformations(edited_attacked_text, None)
        original_text = process_text(original_text)
        # Get the indices of the candidate words to switch out 

        tokens = edited_attacked_text.words
        indicators = [0 for _ in range(len(tokens))]
        original_words = {}  # maps the index of the token in the sentence to the original token
        transformed_words = {}  # maps the index of the token in the sentence to a list of token options (including the original token)
        for attack_text_object in transformed_texts:
            # curr_text = attack_text_object.text
            transformed_tokens = attack_text_object.words
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
        
        # constrained setting
        orig_score = self.get_goal_results(original_text)
        leave_one = []
        if constrain > 0:
            for idx in range(len(legal_actions_mask) - 1):
                i = legal_actions_mask[idx]
                if i > 0:
                    leave_one.append(original_text.replace_word_at_index(idx, self.unk_token))
            leave_res, over = self.get_goal_results(leave_one)
            scores = torch.tensor([result.score for result in leave_res])
            scores -= orig_score
            scores = torch.abs(scores)
            _, idxs = torch.topk(scores, constrain)
            legal_actions_mask = [0] * self.max_num_words_swappable_in_sentence + [1]
            legal_actions_mask[idxs] = 1

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

        curr_state = AttackedText(original_text)
        # print("Num word candidates: ", len(word_candidates), "     Length sentence tokens: ", len(curr_state.words), flush=True)

        curr_state_tokens = curr_state.words
        sentence_embedding, word_embedding, indicator_embedding = self.create_embeddings(curr_state_tokens, word_candidates, indicators)
        prev_word_candidates = copy.deepcopy(word_candidates)
 
        rollout_reward = 0
        for i in range(self.max_length_rollout):

            # action, probs, state_embedding = self.get_action(curr_state_tokens, word_candidates, indicators, legal_actions_mask)

            action, probs = self.get_action(sentence_embedding, word_embedding, indicator_embedding, legal_actions_mask, is_evaluation)

            if action != stop_action:  # if we swap out a word
                index_in_sentence, index_of_word_list = action_to_index[action]

                word_to_replace_with = word_candidates[index_in_sentence][index_of_word_list]

                # Modify word candidates by reinserting the previous word and taking out the word replacement
                word_candidates[index_in_sentence].remove(word_to_replace_with)
                word_candidates[index_in_sentence].append(curr_state_tokens[index_in_sentence])
                word_candidates[index_in_sentence].sort(reverse=False)

                next_state = curr_state.replace_word_at_index(index_in_sentence, word_to_replace_with)
            
            else:
                next_state = AttackedText(curr_state.text)

            r = self.reward_function(curr_state, action, next_state, stop_action)
            # if self._search_over:
            #     print("Search over", self._search_over, "   Reward: ", r)
            rollout_reward += r

            next_state_tokens = next_state.words
            next_sentence_embedding, next_word_embedding, next_indicator_embedding = self.create_embeddings(next_state_tokens, word_candidates, indicators)

            # TODO: Condition adding to buffer on is_evaluation
            if not is_evaluation:
                self.buffer.append(Transition(state=curr_state_tokens, word_candidates=prev_word_candidates, 
                                            action=action, reward=r, next_state=next_state_tokens, 
                                            next_word_candidates=word_candidates, done=int(action == stop_action), 
                                            legal_actions_mask=legal_actions_mask, indicators=indicators))  # legal_actions_mask is same for each rollout

            # Preparing all variables for next iteration
            curr_state = next_state
            curr_state_tokens = curr_state.words
            prev_word_candidates = copy.deepcopy(word_candidates)

            sentence_embedding = next_sentence_embedding
            word_embedding = next_word_embedding
            indicator_embedding = next_indicator_embedding

            # Removed below so that an adversarial example is ONLY returned if the stop_action is actually used
            """
            if self._search_over and action != stop_action:  # if found adversarial example and was not intentional, finish up the episode 
                # Insert to buffer the transition corresponding to stop action
                next_state_tokens = AttackedText(curr_state.text).words 
                self.buffer.append(Transition(state=curr_state_tokens, word_candidates=prev_word_candidates,
                                              action=stop_action, reward=self.lambda_search_success, next_state=next_state_tokens, 
                                              next_word_candidates=word_candidates, done=1, 
                                              legal_actions_mask=legal_actions_mask, indicators=indicators))
                rollout_reward += self.lambda_search_success
                # print("SEARCH OVER! ", self._search_over, "   Adding extra action to list")
                # print(curr_state)
                break
            """

            if action == stop_action:
                break


            # print("Buffer Size: ", len(self.buffer))
            if len(self.buffer) >= self.min_steps_warm_up:
                if len(self.buffer) == self.min_steps_warm_up:
                    print("Beginning training")
                if not is_evaluation:
                    self.update_dqn()
                    self.epsilon = max(self.epsilon_final, self.epsilon - ((self.epsilon_init - self.epsilon_final) / self.epsilon_num_steps))

            
            if len(self.buffer) > self.max_buffer_size:
                self.buffer = self.buffer[-self.max_buffer_size:]
        
        self.episode_returns.append(rollout_reward)
        
        if len(self.episode_returns) % 100 == 0:
            print("Buffer size: ", len(self.buffer))
            print("Average returns past 100 rollouts: ", sum(self.episode_returns[-100:]) / 100, flush=True)
            print("New epsilon value: ", self.epsilon)

        curr_state = self.cache[curr_state.text][0][0]

        return curr_state
        

    def get_action(self, sentence_embedding, word_embedding, indicator_embedding, legal_actions_mask, is_evaluation):
        """
        probs = self.get_action_probs(sentence_embedding, word_embedding, indicator_embedding, legal_actions_mask)
        probs[0, legal_actions_mask] = (1 - self.epsilon) * probs[0, legal_actions_mask]
        probs[0, legal_actions_mask] += self.epsilon / sum(legal_actions_mask)

        # Remember to insert the additional action at the very END corresponding to stop action
        action = probs.multinomial(num_samples=1, replacement=False)[0].item()
        """
        
        output = self.model(sentence_embedding.unsqueeze(0), word_embedding.unsqueeze(0), torch.unsqueeze(indicator_embedding, 0))
        # print("Evaluation: ", is_evaluation)
        if is_evaluation:
            epsilon = 0
        else:
            epsilon = self.epsilon

        if np.random.random() < epsilon:
            probs = np.array(legal_actions_mask) / sum(legal_actions_mask)
            action = np.random.choice(probs.shape[0], p=probs)
            probs = np.zeros(len(legal_actions_mask))
            probs[action] = 1
        else:
            probs = torch.zeros(output.size()[1])
            illegal_action_indicator = [i for i, indicator in enumerate(legal_actions_mask) if indicator == 0]
            output[0, illegal_action_indicator] = -1e9
            action = int(torch.argmax(output, dim=1)[0].item())
            probs[action] = 1
        
 
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

        sentence_embedding = self.embedding(torch.tensor(sentence_embedding_ids, dtype=torch.int32).to(utils.device))

        assert not any([len(w) == 0 for w in word_candidates_sorted])
        word_candidate_ids = [self.query_word_to_id(w) for w in word_candidates_sorted]
        num_pad_tokens = max(self.max_num_words_swappable_in_sentence - len(word_candidate_ids), 0)
        word_candidate_ids = word_candidate_ids + ([self.embedding.padid] * num_pad_tokens)

        word_candidate_embedding = self.embedding(torch.tensor(word_candidate_ids, dtype=torch.int32).to(utils.device))

        # result = torch.concat([torch.flatten(sentence_embedding), torch.flatten(word_candidate_embedding), torch.Tensor(indicators)])

        return sentence_embedding, word_candidate_embedding, torch.Tensor(indicators).to(utils.device)

    
    def get_goal_result_wrapper(self, state):
        # State is an attacked_text object
        if state.text in self.cache:
            goal_result = self.cache[state.text]
            self.cache.pop(state.text)  # remove and reinsert to push it to end of ordered dictionary
        else:
            goal_result = self.get_goal_results([state])
            # s_score = goal_result[0].score 
        self.cache[state.text] = goal_result  # insert to end of ordered dictionary 

        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)  # remove the oldest entry in dictionary 
        return goal_result

    def reward_function(self, s, a, next_s, stop_action):
        # reward an increase in the target output class on the target model 
        s_goal_result, _ = self.get_goal_result_wrapper(s)
        next_s_goal_result, status = self.get_goal_result_wrapper(next_s)

        s_score = s_goal_result[0].score 
        next_s_score = next_s_goal_result[0].score

        self._search_over = (next_s_goal_result[0].goal_status == GoalFunctionResultStatus.SUCCEEDED) and a == stop_action
        # print("SEARCH OVER? ", self._search_over, status, next_s_goal_result[0].goal_status, GoalFunctionResultStatus.SUCCEEDED)

        reward_logit_diff = next_s_score - s_score

        if self._search_over:
            return self.lambda_search_success

        if a == stop_action and not self._search_over:
            return self.constant_reward

        # print("rewards: ", self.lambda_logit_diff * reward_logit_diff)
    
        return self.lambda_logit_diff * reward_logit_diff + self.constant_reward
    
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
            rtg[i] = (rewards[i]) + self.gamma * rtg[i+1] * (1 - dones[i])
        rtg = rtg[:-1]

        rtg = rtg - np.mean(rtg)

        action_probs = self.get_action_probs(torch.stack(sentence_embedding), torch.stack(word_embedding), torch.stack(indicator_embedding), torch.FloatTensor(legal_actions_masks))

        selected_action_probs = torch.gather(action_probs, 1, torch.reshape(torch.tensor(actions), (-1, 1)))

        # Use actions to query gather from action_probs
        # print("Sizes: ", (torch.log(selected_action_probs) * torch.tensor(rtg, dtype=torch.float32).unsqueeze(1)).size())
        loss = -torch.mean(torch.log(selected_action_probs) * torch.tensor(rtg, dtype=torch.float32).unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print("Iteration loss: {}".format(loss.item()))

        self.buffer = []
        return 

    def update_target(self):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(local_param.data)

    def update_dqn(self):
        sentence_embedding = []
        word_embedding = []
        indicator_embedding = []

        next_sentence_embedding = []
        next_word_embedding = []
        next_indicator_embedding = []

        idx = np.random.choice(len(self.buffer), self.batch_size)

        for i in idx:
            t = self.buffer[i]
            curr_sentence_embedding, curr_word_embedding, curr_indicator_embedding = self.create_embeddings(t.state, t.word_candidates, t.indicators)
            sentence_embedding.append(curr_sentence_embedding)
            word_embedding.append(curr_word_embedding)
            indicator_embedding.append(curr_indicator_embedding)
            
            curr_sentence_embedding, curr_word_embedding, curr_indicator_embedding = self.create_embeddings(t.next_state, t.next_word_candidates, t.indicators)
            next_sentence_embedding.append(curr_sentence_embedding)
            next_word_embedding.append(curr_word_embedding)
            next_indicator_embedding.append(curr_indicator_embedding)
            
        actions = [self.buffer[i].action for i in idx]
        rewards = [self.buffer[i].reward for i in idx]
        dones = [self.buffer[i].done for i in idx]
        legal_actions_masks = torch.tensor([self.buffer[i].legal_actions_mask for i in idx]).to(utils.device)

        q_values = self.model(torch.stack(sentence_embedding), torch.stack(word_embedding), torch.stack(indicator_embedding))
        q_values_chosen = torch.gather(q_values, 1, torch.tensor(actions).to(utils.device).unsqueeze(1))

        next_q_values_target = self.target_model(torch.stack(next_sentence_embedding), torch.stack(next_word_embedding), torch.stack(next_indicator_embedding))

        q_values_next = self.model(torch.stack(next_sentence_embedding), torch.stack(next_word_embedding), torch.stack(next_indicator_embedding)).detach()
        filtered_q_values_next = torch.where(legal_actions_masks == 1, q_values_next, -1e9)
        next_a = torch.argmax(filtered_q_values_next, dim=1)
        
        # filtered_next_q_values = torch.where(legal_actions_masks == 1, next_q_values_target, -1e9)
        # next_max_q_values = torch.max(filtered_next_q_values, dim=1)[0]

        next_max_q_values = torch.gather(next_q_values_target, 1, next_a.unsqueeze(1)).squeeze()

        # print(next_a.unsqueeze(1).size(), next_q_values_target.size(), next_max_q_values.size(), torch.tensor(rewards).size())
        assert torch.tensor(rewards).size() == next_max_q_values.size()
        target = torch.tensor(rewards).to(utils.device) + self.gamma * next_max_q_values 
        target = target.unsqueeze(1)

        loss_function = torch.nn.MSELoss()
        target.detach()

        assert q_values_chosen.size() == target.size()
        loss = loss_function(q_values_chosen, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # print("DQN loss: ", loss)

        self.updates_so_far += 1
        if self.updates_so_far % self.update_target_every == 0:
            self.update_target()

