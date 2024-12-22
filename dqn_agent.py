import os
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential, clone_model, load_model
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Input
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

        # Build the main model and target model
        self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())  # Initialize with same weights

        # Ensure checkpoint directory exists
        os.makedirs("checkpoints", exist_ok=True)

    def _build_model(self):
        """Build the LSTM-based neural network model."""
        model = Sequential()
        model.add(Input(shape=(1, self.state_size)))
        model.add(LSTM(64, activation='relu', return_sequences=True))
        model.add(BatchNormalization())
        model.add(LSTM(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        """Update the target network weights."""
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        """Choose an action based on epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore: Random action
        
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # Exploit: Choose action with highest Q-value

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
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
                target += self.gamma * np.amax(self.target_model.predict(next_state)[0])
            
            target_f = self.model.predict(state)
            target_f[0][action] = target
            
            # Train the main network on this sample
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Decay epsilon (exploration rate)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_checkpoint(self, episode):
        """Save the trained model as a checkpoint."""
        filepath = f"checkpoints/dqn_checkpoint_episode_{episode}.h5"
        
        try:
            os.makedirs("checkpoints", exist_ok=True)  # Ensure directory exists
            self.model.save(filepath)
            print(f"[INFO] Checkpoint saved: {filepath}")
        
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {e}")

    def load_checkpoint(self):
        """Load the latest checkpoint from the checkpoints directory."""
        
        try:
            checkpoints = [f for f in os.listdir("checkpoints") if f.endswith(".h5")]
            
            if not checkpoints:
                print("[INFO] No checkpoints found. Starting fresh.")
                return False
            
            # Sort checkpoints by episode number and load the latest one
            checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            latest_checkpoint = checkpoints[-1]
            
            filepath = os.path.join("checkpoints", latest_checkpoint)
            self.model = load_model(filepath)
            
            print(f"[INFO] Loaded checkpoint: {filepath}")
            return True
        
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            return False


# training loop
def __init__(self, state_size=None):
    self.state_size = state_size or len(self.get_feature_names())
    self.action_size = 3  # Buy, Sell, Hold
    self.memory = deque(maxlen=2000)
    self.gamma = 0.95
    self.epsilon = 1.0
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.learning_rate = 0.001
    
    # Initialize models
    self.model = self._build_model()
    self.target_model = clone_model(self.model)
    self.target_model.set_weights(self.model.get_weights())


    # Try loading the latest checkpoint
    def train(self, episodes=100, batch_size=32):
        self.load_checkpoint()  # Load previous training if available
    
        for episode in range(1, episodes + 1):
            state = self.get_initial_state()  # Replace with actual state initialization
            episode_reward = 0
            step = 0
        
            while not self.is_episode_done(step):
                action = self.act(state)
                next_state, reward, done = self.step(action)  # Implement actual environment step
            
                self.remember(state, action, reward, next_state, done)
                self.replay(batch_size)
            
                if step % 100 == 0:  # Update target network periodically
                    self.update_target_model()
                
                state = next_state
                episode_reward += reward
                step += 1
            
            if episode % 10 == 0:
                self.save_checkpoint(episode)
                print(f"Episode: {episode}, Reward: {episode_reward}, Epsilon: {self.epsilon:.2f}")


        print("[INFO] Training complete.")
