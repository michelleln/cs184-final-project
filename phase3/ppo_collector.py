import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        print(f"Creating PolicyNetwork with state_dim={state_dim}, action_dim={action_dim}")
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        print(f"Input shape to PolicyNetwork: {x.shape}")
        return self.network(x)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),  # Matching architecture with policy network
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        return self.network(x)

class PPOCollector:
    def __init__(self, env, learning_rate=0.0003, gamma=0.99, clip_epsilon=0.2, c1=1, c2=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get dimensions
        self.state_dim = env.get_state_space_size()
        print(f"Initializing PPOCollector with state_dim={self.state_dim}")
        self.action_dim = env.get_action_space_size()
        print(f"Action dim={self.action_dim}")
        
        # Initialize networks
        self.policy_network = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.value_network = ValueNetwork(self.state_dim).to(self.device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)

        # Add gradient clipping parameters
        self.max_grad_norm = 0.5
        self.value_coef = 0.5
        self.entropy_coef = 0.01

    def get_action(self, state):
        """Select action based on current policy"""
        print(f"get_action input state shape: {state.shape}")
        state_tensor = torch.FloatTensor(state)
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
        print(f"get_action processed state shape: {state_tensor.shape}")
        
        with torch.no_grad():
            action_probs = self.policy_network(state_tensor).squeeze(0)
            print(f"action_probs shape: {action_probs.shape}")
            
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()

        
    def update_policy(self, states, actions, old_log_probs, advantages, returns):
        # Convert to tensors and ensure proper shapes
        states = torch.FloatTensor(states).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(0)
        
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Get current action probabilities and values
        action_probs = self.policy_network(states)
        dist = Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Get state values and ensure proper shape
        values = self.value_network(states).squeeze()
        if len(values.shape) == 0:
            values = values.unsqueeze(0)
        
        # Calculate ratios and surrogate objectives
        ratios = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
        
        # Calculate losses
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * ((returns - values) ** 2).mean()
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Update networks
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()
        
        # Clip gradients
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.value_network.parameters(), self.max_grad_norm)
        
        self.policy_optimizer.step()
        self.value_optimizer.step()
        
        return policy_loss.item(), value_loss.item(), entropy.item()

    def collect_trajectory(self, max_steps=1000):
        """Collect a single trajectory using current policy"""
        states, actions, rewards, log_probs = [], [], [], []
        total_reward = 0
        state = self.env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            action, log_prob = self.get_action(state)
            next_state, reward, done = self.env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            state = next_state
            total_reward += reward
            steps += 1
                
        return states, actions, rewards, log_probs, total_reward

    def train(self, n_epochs=100, n_steps=1000, batch_size=64, gamma=0.99, gae_lambda=0.95):
        rewards_history = []
        
        for epoch in range(n_epochs):
            # Collect trajectory
            states, actions, rewards, log_probs, total_reward = self.collect_trajectory(n_steps)
            rewards_history.append(total_reward)
            
            # Ensure states is a proper batch
            states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            
            # Calculate values for all states
            with torch.no_grad():
                values = self.value_network(states_tensor).squeeze().cpu().numpy()
                # Ensure values is 1D array matching rewards length
                if len(values.shape) == 0:
                    values = np.array([values])
            
            # Convert rewards to numpy array if not already
            rewards = np.array(rewards)
            
            # Initialize advantage and returns arrays
            advantages = np.zeros_like(rewards, dtype=np.float32)
            returns = np.zeros_like(rewards, dtype=np.float32)
            
            # GAE calculation
            last_gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]
                
                # Calculate TD error and advantage
                delta = rewards[t] + gamma * next_value - values[t]
                advantages[t] = last_gae = delta + gamma * gae_lambda * last_gae
            
            # Calculate returns
            returns = advantages + values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Convert to batches for training
            n_samples = len(states)
            batch_indices = np.arange(n_samples)
            
            # Update policy multiple times
            n_batches = max(n_samples // batch_size, 1)
            for _ in range(n_batches):
                # Sample random batch
                batch_idx = np.random.choice(batch_indices, size=min(batch_size, n_samples), replace=False)
                
                # Get batch data
                batch_states = np.array(states)[batch_idx]
                batch_actions = np.array(actions)[batch_idx]
                batch_log_probs = np.array(log_probs)[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Update policy
                policy_loss, value_loss, entropy = self.update_policy(
                    batch_states,
                    batch_actions,
                    batch_log_probs,
                    batch_advantages,
                    batch_returns
                )
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Total Reward: {total_reward:.2f}, Policy Loss: {policy_loss:.3f}")
        
        return rewards_history


def run_ppo_test():
    from collection_draft import PACKPOOL
    import matplotlib.pyplot as plt
    
    # Initialize environment and agent
    env = PACKPOOL(num_packs=5)
    
    # Set target collection
    target_cards = ["mushroom", "mushroom", "toadFoil", "turtle", "turtle", 
                   "turtleFoil", "firedragon", "firedragon", "originFoil"]
    env.set_target_collection(target_cards)
    
    # Debug dimensions
    state = env.reset()
    print(f"State shape: {state.shape}")
    print(f"State space size: {env.get_state_space_size()}")
    print(f"Action space size: {env.get_action_space_size()}")
    
    agent = PPOCollector(env)
    
    # Test state processing
    test_action, test_log_prob = agent.get_action(state)
    print(f"Test action: {test_action}")
    print(f"Test log prob: {test_log_prob}")
    
    print("Starting training...")
    rewards = agent.train(n_epochs=100)
    print("Training complete!")
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('PPO Learning Curve')
    plt.grid(True)
    plt.show()


run_ppo_test()
