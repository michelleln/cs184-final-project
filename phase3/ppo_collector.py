import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """This neural network maps states to action probabilities. It is used to sample
    actions during trajectory collection and is updated via the PPO objective to learn 
    the optimal policy."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
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
            x = x.unsqueeze(0)
        return self.network(x)

class ValueNetwork(nn.Module):
    """This neural network estimates the expected return (value) from each state. It
    is used as a baseline in calculating advantages"""
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return self.network(x)
    
class PPOCollector:
    def __init__(self, env, lambda_=0.01, learning_rate=0.0003, gamma=0.99):
        self.env = env
        self.lambda_ = lambda_  # KL penalty coefficient
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get dimensions
        self.state_dim = env.get_state_space_size()
        self.action_dim = env.get_action_space_size()
        
        # Initialize networks
        self.policy_network = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.value_network = ValueNetwork(self.state_dim).to(self.device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)

    def compute_kl_divergence(self, states, old_policy_probs, new_policy_probs):
        """Compute KL divergence between old and new policies following eq 6.47"""
        kl_div = (old_policy_probs * (torch.log(old_policy_probs) - torch.log(new_policy_probs))).sum(dim=1)
        return kl_div.mean()

    def compute_advantages(self, states, rewards):
        """Compute advantage estimates for PPO"""
        with torch.no_grad():
            values = self.value_network(states).squeeze()
            if len(values.shape) == 0:
                values = values.unsqueeze(0)

            next_values = torch.cat([values[1:], torch.tensor([0.0])])
            advantages = rewards + self.gamma * next_values - values
        return advantages

    def compute_ppo_objective(self, states, actions, old_probs, advantages):
        """Compute PPO objective following eq 6.50"""
        # Get new policy probabilities
        new_probs = self.policy_network(states)
        dist = Categorical(new_probs)
        new_log_probs = dist.log_prob(actions)
        old_log_probs = torch.log(old_probs)

        # Compute probability ratio (π_θ / π_k)
        ratio = torch.exp(new_log_probs - old_log_probs)

        # First term: importance weighted advantages
        advantage_estimate = ratio * advantages

        # Second term: KL penalty (following eq 6.47)
        kl_penalty = self.lambda_ * torch.clamp(torch.log(1.0 / new_probs[range(len(actions)), actions]), min=-10, max=10)

        # Combined PPO objective (eq 6.47)
        objective = advantage_estimate - kl_penalty

        return objective.mean()

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
        
        # Calculate ratios and advantage estimate in the PPO objective 
        ratios = torch.exp(new_log_probs - old_log_probs)
        advantage_estimate1 = ratios * advantages
        advantage_estimate2 = torch.clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
        # do this to avoid NaN values when ratio/probabilities too small
        
        # Calculate losses
        policy_loss = -torch.min(advantage_estimate1, advantage_estimate2).mean()
        value_loss = 0.5 * ((returns - values) ** 2).mean()
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Update networks
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()
        
        # Clip gradients to avoid NaN values when gradient too small
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.value_network.parameters(), self.max_grad_norm)
        
        self.policy_optimizer.step()
        self.value_optimizer.step()
        
        return policy_loss.item(), value_loss.item(), entropy.item()

    def collect_trajectory(self, max_steps=1000):
        """Collect a trajectory following current policy"""
        states, actions, rewards, probs = [], [], [], []
        state = self.env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < max_steps:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).to(self.device)
            
            # Get action probabilities
            with torch.no_grad():
                action_probs = self.policy_network(state_tensor)
                dist = Categorical(action_probs)
                action = dist.sample()
            
            # Take action in environment
            next_state, reward, done = self.env.step(action.item())
            
            # Store transition
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            probs.append(action_probs[0, action].item())
            
            state = next_state
            total_reward += reward
            steps += 1
        
        return states, actions, rewards, probs, total_reward


    def train(self, n_epochs=40):
        """Train using PPO following lecture notes algorithm"""
        for epoch in range(n_epochs):
            # Collect trajectory using current policy (π_k)
            states, actions, rewards, old_probs, total_reward = self.collect_trajectory()
            
            # Convert to tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            old_probs = torch.FloatTensor(old_probs).to(self.device)
            
            # Compute advantages
            advantages = self.compute_advantages(states, rewards)
            
            # Update policy (θ^(k+1) ← arg max l^k(θ)) following eq 6.51
            self.policy_optimizer.zero_grad()
            objective = self.compute_ppo_objective(states, actions, old_probs, advantages)
            (-objective).backward()  # Negative because we're maximizing
            self.policy_optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Total Reward: {total_reward:.2f}, learning rate={self.learning_rate}")
                
        return total_reward



def run_ppo_test():
    from phase3.collection_draft_ppo import PACKPOOL
    import matplotlib.pyplot as plt
    
    # Initialize environment and agent
    env = PACKPOOL(num_packs=5)
    
    # Set target collection
    target_cards = ["mushroom", "mushroom", "toadFoil", "turtle", "turtle", 
                   "turtleFoil", "firedragon", "firedragon", "originFoil"]
    env.set_target_collection(target_cards)
    
    agent = PPOCollector(env)
    
    # Train and track rewards
    rewards = []
    for i in range(40):  # 40 epochs
        reward = agent.train(n_epochs=1)  # Train for one epoch
        rewards.append(reward)
        if i % 10 == 0:
            print(f"Training iteration {i}, Reward: {reward}")
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'PPO Learning Curve, learning rate={agent.learning_rate}')
    plt.grid(True)
    plt.show()


run_ppo_test()
