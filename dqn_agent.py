import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


class DQNLSTMAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Replay memory
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # StandardScaler for state normalization
        self.scaler = StandardScaler()

        # Build the LSTM-based neural network model
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=(1, self.state_size), activation='relu', return_sequences=True))
        model.add(LSTM(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def act(self, state):
        """Choose an action based on epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore: Random action
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # Exploit: Choose action with highest Q-value

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.
        """
        # Normalize states
        normalized_state = self.scaler.transform(state.reshape(1, -1)).reshape(1, 1, -1)
        normalized_next_state = self.scaler.transform(next_state.reshape(1, -1)).reshape(1, 1, -1)

        self.memory.append((normalized_state, action, reward, normalized_next_state, done))

    def replay(self, batch_size):
        """Train the model using a batch of experiences from replay memory."""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
