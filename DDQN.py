import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.layers import LSTMCell

class DuelingDoubleDeepQNetwork:

    def __init__(self,
                 n_actions,                  # the number of actions
                 n_features,
                 n_lstm_features,
                 n_time,
                 learning_rate = 0.01,
                 reward_decay = 0.9,
                 e_greedy = 0.99,
                 replace_target_iter = 200,  # each 200 steps, update target net
                 memory_size = 500,  # maximum of memory
                 batch_size=32,
                 e_greedy_increment= 0.00025,
                 n_lstm_step = 10,
                 dueling = True,
                 double_q = True,
                 hidden_units_l1 = 20,
                 N_lstm = 20):

        self.n_actions = n_actions
        self.n_features = n_features
        self.n_time = n_time
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size    # select self.batch_size number of time sequence for learning
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.dueling = dueling
        self.double_q = double_q
        self.learn_step_counter = 0
        self.hidden_units_l1 = hidden_units_l1

        # lstm
        self.N_lstm = N_lstm
        self.n_lstm_step = n_lstm_step       # step_size in lstm
        self.n_lstm_state = n_lstm_features  # [fog1, fog2, ...., fogn, M_n(t)]

        # initialize zero memory np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
        self.memory = np.zeros((self.memory_size, self.n_features + 1 + 1
                                    + self.n_features + self.n_lstm_state + self.n_lstm_state))

        # consist of [target_net, evaluate_net]
        self._build_net()

        # replace the parameters in target net
        self.replace_target_op = lambda: self.target_net.set_weights(self.eval_net.get_weights())

        # Removed TensorFlow 1.x session management, not needed in TensorFlow 2.x

        self.reward_store = list()
        self.action_store = list()
        self.delay_store = list()
        self.energy_store = list()

        self.lstm_history = deque(maxlen=self.n_lstm_step)
        for ii in range(self.n_lstm_step):
            self.lstm_history.append(np.zeros([self.n_lstm_state]))

        self.store_q_value = list()

        # Removed TensorFlow 1.x Saver and get_collection, not needed in TensorFlow 2.x

    def _build_net(self):
        tf.compat.v1.reset_default_graph()

        # input for eval_net
        self.s = tf.keras.Input(shape=(self.n_features,), dtype=tf.float32, name='s')  # state (observation)
        self.lstm_s = tf.keras.Input(shape=(self.n_lstm_step, self.n_lstm_state), dtype=tf.float32, name='lstm1_s')
        self.q_target = tf.keras.Input(shape=(self.n_actions,), dtype=tf.float32, name='Q_target')  # q_target

        # input for target_net
        self.s_ = tf.keras.Input(shape=(self.n_features,), dtype=tf.float32, name='s_')
        self.lstm_s_ = tf.keras.Input(shape=(self.n_lstm_step, self.n_lstm_state), dtype=tf.float32, name='lstm1_s_')

        # generate EVAL_NET, update parameters
        self.eval_net = EvalNet(self.n_actions, self.n_features, self.n_lstm_step, self.n_lstm_state,
                                self.hidden_units_l1, self.N_lstm, self.dueling)
        self.q_eval = self.eval_net([self.s, self.lstm_s])

        # generate TARGET_NET
        self.target_net = EvalNet(self.n_actions, self.n_features, self.n_lstm_step, self.n_lstm_state,
                                  self.hidden_units_l1, self.N_lstm, self.dueling)
        self.q_next = self.target_net([self.s_, self.lstm_s_])

        # Compile the model with RMSprop optimizer and custom loss
        self.eval_net.compile(optimizer=tf.keras.optimizers.RMSprop(self.lr), loss=CustomSquaredError())

    def store_transition(self, s, lstm_s,  a, r, s_, lstm_s_):
        # RL.store_transition(observation,action,reward,observation_)
        # hasattr(object, name), if object has name attribute
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # store np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
        transition = np.hstack((s, [a, r], s_, lstm_s, lstm_s_))  # stack in horizontal direction

        # if memory overflows, replace old memory with new one
        index = self.memory_counter % self.memory_size
        # print(transition)
        self.memory[index, :] = transition
        self.memory_counter += 1

    def update_lstm(self, lstm_s):

        self.lstm_history.append(lstm_s)

    def choose_action(self, observation):
        # the shape of the observation (1, size_of_observation)
        # x1 = np.array([1, 2, 3, 4, 5]), x1_new = x1[np.newaxis, :], now, the shape of x1_new is (1, 5)

        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:

            # lstm only contains history, there is no current observation
            lstm_observation = np.array(self.lstm_history)

            actions_value = self.eval_net.predict([observation, lstm_observation.reshape(1, self.n_lstm_step, self.n_lstm_state)])

            self.store_q_value.append({'observation': observation, 'q_value': actions_value})

            action = np.argmax(actions_value)

        else:


            if np.random.randint(0,100) < 25:
                action = np.random.randint(1, self.n_actions)
            else:
                action = 0


        return action

    def learn(self):

        # check if replace target_net parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            # Directly set the weights of the target network to the weights of the evaluation network
            self.replace_target_op()
            print('\ntarget_params_replaced')

        # randomly pick [batch_size] memory from memory np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size - self.n_lstm_step, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter - self.n_lstm_step, size=self.batch_size)\

        #  transition = np.hstack(s, [a, r], s_, lstm_s, lstm_s_)
        batch_memory = self.memory[sample_index, :self.n_features+1+1+self.n_features]
        lstm_batch_memory = np.zeros([self.batch_size, self.n_lstm_step, self.n_lstm_state * 2])
        for ii in range(len(sample_index)):
            for jj in range(self.n_lstm_step):
                lstm_batch_memory[ii,jj,:] = self.memory[sample_index[ii]+jj,
                                              self.n_features+1+1+self.n_features:]

        # obtain q_next (from target_net) (to q_target) and q_eval (from eval_net)
        # minimize（target_q - q_eval）^2
        # q_target = reward + gamma * q_next
        # in the size of bacth_memory
        # q_next, given the next state from batch, what will be the q_next from q_next
        # q_eval4next, given the next state from batch, what will be the q_eval4next from q_eval
        q_next = self.target_net.predict([batch_memory[:, -self.n_features:], lstm_batch_memory[:,:,self.n_lstm_state:]])
        q_eval4next = self.eval_net.predict([batch_memory[:, -self.n_features:], lstm_batch_memory[:,:,self.n_lstm_state:]])
        q_eval = self.eval_net.predict([batch_memory[:, :self.n_features], lstm_batch_memory[:,:,:self.n_lstm_state]])
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)  # action with a single value (int action)
        reward = batch_memory[:, self.n_features + 1]  # reward with a single value

        # update the q_target at the particular batch at the correponding action
        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)
            selected_q_next = q_next[batch_index, max_act4next]
        else:
            selected_q_next = np.max(q_next, axis=1)

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        # Perform training using the fit method
        self.eval_net.fit(x=[batch_memory[:, :self.n_features], lstm_batch_memory[:, :, :self.n_lstm_state]],
                          y=q_target, batch_size=self.batch_size, verbose=0)

        # gradually increase epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def do_store_reward(self, episode, time, reward):
        while episode >= len(self.reward_store):
            self.reward_store.append(np.zeros([self.n_time]))
        self.reward_store[episode][time] = reward

    def do_store_action(self,episode,time, action):
        while episode >= len(self.action_store):
            self.action_store.append(- np.ones([self.n_time]))
        self.action_store[episode][time] = action

    def do_store_delay(self, episode, time, delay):
        while episode >= len(self.delay_store):
            self.delay_store.append(np.zeros([self.n_time]))
        self.delay_store[episode][time] = delay

    def do_store_energy(self, episode, time, energy, energy2, energy3, energy4):

        fog_energy = 0
        for i in range(len(energy3)):
            if energy3[i] != 0:
                fog_energy = energy3[i]


        idle_energy = 0
        for i in range(len(energy4)):
            if energy4[i] != 0:
                idle_energy = energy4[i]

        while episode >= len(self.energy_store):
            self.energy_store.append(np.zeros([self.n_time]))
        self.energy_store[episode][time] = energy + energy2 + fog_energy + idle_energy



    def Initialize(self,iot):
        self.load_model(iot)


    def load_model(self,iot):
        latest_ckpt = tf.train.latest_checkpoint("./models/500/"+str(iot)+"_X_model")

        print(latest_ckpt, "_____+______________________________________________")
        if latest_ckpt is not None:
            self.eval_net = tf.keras.models.load_model(latest_ckpt)

class EvalNet(tf.keras.Model):
    def __init__(self, n_actions, n_features, n_lstm_step, n_lstm_state, hidden_units_l1, N_lstm, dueling, **kwargs):
        super(EvalNet, self).__init__(**kwargs)
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_lstm_step = n_lstm_step
        self.n_lstm_state = n_lstm_state
        self.hidden_units_l1 = hidden_units_l1
        self.N_lstm = N_lstm
        self.dueling = dueling

        self.lstm_layer = tf.keras.layers.LSTM(self.N_lstm, return_sequences=True, return_state=True)
        self.l1 = tf.keras.layers.Dense(hidden_units_l1, activation='relu')
        self.l12 = tf.keras.layers.Dense(hidden_units_l1, activation='relu')
        if self.dueling:
            self.value = tf.keras.layers.Dense(1)
            self.advantage = tf.keras.layers.Dense(self.n_actions)
        else:
            self.q_values = tf.keras.layers.Dense(self.n_actions)

    def call(self, inputs):
        s, lstm_s = inputs
        lstm_output, _, _ = self.lstm_layer(lstm_s)
        lstm_output_reduced = tf.reshape(lstm_output[:, -1, :], shape=[-1, self.N_lstm])
        l1_output = self.l1(tf.concat([lstm_output_reduced, s], 1))
        l12_output = self.l12(l1_output)

        if self.dueling:
            V = self.value(l12_output)
            A = self.advantage(l12_output)
            out = V + (A - tf.reduce_mean(A, axis=1, keepdims=True))
        else:
            out = self.q_values(l12_output)
        return out

class CustomSquaredError(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='custom_squared_error'):
        super(CustomSquaredError, self).__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)
