import numpy as np
import torch
from torch.nn.functional import softmax

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod

Transition = collections.namedtuple(
    "Transition",
    "state action reward next_state done legal_actions_mask")


class RLWordSwap(SearchMethod): 
    def __init__(self, lr, gamma, batch_size, max_num_words_swap):

        self.model = 
        self.buffer = []

        self.optimizer = torch.optim.Adam(lr=lr)

        self.max_length_rollout = 50
        self.batch_size = batch_size

        # Hyperparameters for reward function 
        self.lambda_logit_diff = 1
        self.lambda_search_success = 5
        self.constant_reward = -.1
        return 

    def 
    
    def perform_search(self, initial_result):
        # Initial result is the input 
        self._search_over = False 
        original_text = initial_result.attacked_text
        curr_state = initial_result.attacked_text

        transformed_texts = self.get_transformations(curr_state, original_text)
        # Get the indices of the candidate words to switch out 
        tokens = original_text.split(' ')
        indicators = [0 for _ in range(len(tokens))]
        index_indicators = []
        original_word = []
        transformed_word = []
        for text in transformed_texts:
            transformed_tokens = text.split(' ')
            for j in len(tokens):
                if tokens[j] != transformed_tokens:
                    indicators[j] = 1
                    original_word.append(tokens[j])
                    transformed_word.append(transformed_tokens[j])
                    index_indicators.append(j)     

        for i in range(self.max_length_rollout):
            word_candidates = []
            for j,k in enumerate(index_indicators): 
                if curr_state[k] == original_word[j]:
                    word_candidates.append(transformed_word[j])
                else:
                    word_candidates.append(original_word[j])
            
            action, probs = self.get_action(curr_state, word_candidates, indicators)

            next_state = curr_state.copy()
            next_state[index_indicators[action]] = word_candidates[action]

            r = self.reward_function(curr_state, action, next_state)

            self.buffer.append(Transition(state=curr_state, action=action, reward=r, next_state=next_state, done=self._search_over))

        if len(self.buffer) > self.batch_size:
            self.update()

    def get_action(self, curr_attacked_text, word_candidates, indicators):
        # curr_attacked_text is going 

        # TODO: Figure out embeddings. We should let word_candidates be a list of strings but then how do we convert to embeddings? Model side of things?
        output = self.model.forward(curr_attacked_text, word_candidates, indicators)

        probs = torch.softmax(output)

        action = torch.sample(probs, dim=1)

        return action, probs

    def reward_function(self, s, a, next_s):
        # reward an increase in the target output class on the target model 
        original_results, _ = self.get_goal_results([s])
        new_results, self._search_over = self.get_goal_results([next_s])
        
        reward_logit_diff = new_results[0].score - original_results[0].score 
        reward_success = self.lambda_search_success if self._search_over else 0
    
        return self.lambda_logit_diff * reward_logit_diff + reward_success + self.constant_reward

    def update(self)
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

    