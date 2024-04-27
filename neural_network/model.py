import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import grpc

DEVICE = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
# DEVICE = T.device('cpu')

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10_000, gamma=0.5)
        self.loss = nn.MSELoss()
        self.device = DEVICE

        self.non_linearity = nn.Mish()
        self.to(self.device)
        
        print("Q-Network Parameters:")
        print(self.fc1)
        print(self.fc2)
        print(self.fc3)

    def forward(self, state: T.Tensor):
        x = self.dropout1(self.non_linearity(self.fc1(state)))
        x = self.dropout2(self.non_linearity(self.fc2(x)))
        actions = self.fc3(x)

        return actions
        
    def display_parameters(self):
        print("Q-Network Parameters:")
        print("FC1 parameters:")
        for name, param in self.fc1.named_parameters():
            print(name, param.data)
        print("FC2 parameters:")
        for name, param in self.fc2.named_parameters():
            print(name, param.data)
        print("FC3 parameters:")
        for name, param in self.fc3.named_parameters():
            print(name, param.data)


#Agent class is responsible for managing the Deep Q-Network (DQN) agent, including initializing its parameters and memory buffers.
class Agent:
    def __init__(
            self,
            gamma,
            batch_size,
            n_actions,
            epsilon_greedy,
            epsilon,
            eps_end,
            eps_decay,
            input_dims,
            lr,
            max_mem_size = 1_000_000
            
    ) -> None:
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]
        
        self.epsilon_greedy = epsilon_greedy
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.input_dims = input_dims
        self.lr = lr
        self.max_mem_size = max_mem_size
        

        self.mem_cntr = 0
        self.Q_eval = DeepQNetwork(lr=self.lr, input_dims=input_dims, n_actions=n_actions, fc1_dims=128, fc2_dims=512, )
        
        self.Q_eval.display_parameters()

        self.state_memory = np.zeros((self.max_mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros_like(self.state_memory, dtype=np.float32)

        self.action_memory = np.zeros(self.max_mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.max_mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.max_mem_size, dtype=np.bool_)
        self.counter = 0


#The store_transition method of the Agent class is responsible for storing a
#transition tuple (state, action, reward, new_state, done) in the agent's replay memory.

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.max_mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1
        
        print("Stored Transition:")
        print("State:", state)
        print("Action:", action)
        print("Reward:", reward)
        print("Next State:", state_)
        print("Done:", done)

#choose_action method of the Agent class is responsible for selecting an action based on the current observation (state)
    def choose_action(self, observation):
       
        # EPSILON GREEDY STRATEGY
        decay_step = 0
        if self.epsilon_greedy:
        # Here we'll use an improved version of our epsilon greedy strategy for Q-learning
            explore_probability = self.eps_end + (self.epsilon - self.eps_end) * np.exp(-self.eps_decay * decay_step)
            print("Current Epsilon:", self.epsilon)
            #decay_step +=1
            
        
        # OLD EPSILON STRATEGY
        else:
            if self.epsilon > self.eps_end:
                self.epsilon *= (1-self.eps_decay)
                explore_probability = self.epsilon
                
        print("Probability of Exploration:", explore_probability)

        if explore_probability > np.random.rand():
        # Make a random action (exploration)
            print("EXPLORATION")
            return random.randrange(len(self.action_space))
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            # Take the biggest Q value (= the best action)
            print("EXPLOITATION")
            state = T.tensor(observation).clone().detach().to(self.Q_eval.device)
            actions = self.Q_eval(state)
            return T.argmax(actions)
    
      #  print("Value of np.random.random()=", np.random.random())
      #  print("Value of eps= ", self.epsilon)
        
      #  if np.random.random() > self.epsilon:
      #      print("EXPLORATION")
      #      action = np.random.choice(self.action_space)
            
      #  else:
      #      print("EXPLOITATION")
      #      state = T.tensor(observation).clone().detach().to(self.Q_eval.device)
      #      actions = self.Q_eval(state)
      #      action = T.argmin(actions).item()
      #  print("Chosen Action:", action)

       # return action
        
                                

    # updating the Q-network's parameters based on the gathered experiences from the replay memory
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.max_mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        # new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        # terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)


        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval(state_batch)[batch_index, action_batch]
        # q_next = self.Q_eval(new_state_batch)
        # q_next[terminal_batch] = 0.0

        q_target = reward_batch # + self.gamma
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.Q_eval.scheduler.step()
        self.counter += 1
        
        print("Learning Update:")
        print("Batch Loss:", loss.item())
        print("Learning Rate:", self.Q_eval.optimizer.param_groups[0]['lr'])
