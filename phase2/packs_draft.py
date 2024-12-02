import numpy as np
import random

cglobal_list = ["mushroom", "moth", "bramble", "mantis", "beetle", "pollenpuff", "crown", "goat", "fox", "horse", "magma", "anteater", "lizard", "centipede", "duck", 
                "jellyfish", "seal", "clam", "crab", "seahorse", "goldfish", "swan", "cucumber", "triggerfish", "snowmoth", "ball", "electric", "jolt", "zebra", 
                "frill", "urchin", "fairy", "slow", "mime", "model", "stamp", "golem", "pangolin", "mole", "ape", "boomerang", "kick", "punch", "drill", "martial", 
                "octopus", "snake", "bat", "maw", "blades", "rat", "sparrow", "cat", "leek", "trio", "tongue", "egg", "bull", "data", "chinchilla", "sheep"]
uglobal_list = ["toad", "butterfly", "bee", "flower", "pitcher", "tree", "firedragon", "dog", "flare", "firebird", "turtle", "frog", "starfish", "carp", "nessie", 
                "vapor", "nautilus", "icebird", "ninja", "mouse", "magnet", "zapbird", "eel", "spoons", "ghost", "hypnosis", "experiment", "lady", "wrestler", 
                "boulder", "rocksnake", "horseshoe", "queen", "king", "sludge", "gas", "boltnut", "dragon", "pidgeon", "song", "parent", "mimic", "pterodactyl", 
                "bear"]
fglobal_list = ["toad", "tree", "firedragon", "dog", "firebird", "turtle", "starfish", "icebird", "mouse", "zapbird", "ghost", "experiment", "wrestler", "boomerang", 
                "song", "origin"]

class PACKPOOL:
    def __init__(self, num_packs=5):
        """
        Initialize the PACKPOOL with a fixed number of packs and set market values for cards.
        
        Parameters:
        - num_packs (int): Number of packs in the pool.
        """
        self.num_packs = num_packs

        # Define fixed market values for all cards with constraints
        self.market_values = self.set_market_values()

        # Initialize packs
        self.packs = [
            {
                "common_list": random.sample(cglobal_list, 38),
                "uncommon_list": random.sample(uglobal_list, 14),
                "foil_list": random.sample(fglobal_list, 6),
                "common_probabilities": None,
                "uncommon_probabilities": None,
                "foil_probabilities": None
            }
            for _ in range(self.num_packs)
        ]

        # Assign probabilities for each pack
        for pack in self.packs:
            pack["common_probabilities"] = self.assign_probabilities(pack["common_list"])
            pack["uncommon_probabilities"] = self.assign_probabilities(pack["uncommon_list"])
            pack["foil_probabilities"] = self.assign_probabilities(pack["foil_list"])

    def set_market_values(self):
        """
        Assign fixed market values to all cards in the global lists with constraints:
        - Common cards: 1 to 5
        - Uncommon cards: 6 to 10
        - Foil cards: 11 to 20
        
        Returns:
        - dict: A dictionary with card names as keys and their fixed market values as values.
        """
        common_values = {card: np.random.uniform(1, 5) for card in cglobal_list}
        uncommon_values = {card: np.random.uniform(6, 10) for card in uglobal_list}
        foil_values = {card: np.random.uniform(11, 20) for card in fglobal_list}
        return {**common_values, **uncommon_values, **foil_values}

    def assign_probabilities(self, card_list):
        """
        Assign probabilities to a given card list.
        
        Parameters:
        - card_list (list): List of cards to assign probabilities to.
        
        Returns:
        - dict: Dictionary with cards as keys and probabilities as values.
        """
        raw_probabilities = np.random.uniform(1, 10, size=len(card_list))
        normalized_probabilities = raw_probabilities / np.sum(raw_probabilities)
        return dict(zip(card_list, normalized_probabilities))
    
    def open_pack(self, pack_id):
        """
        Simulate opening a pack and calculate the total market value of the cards.
        
        Parameters:
        - pack_id (int): Index of the pack to open.
        
        Returns:
        - float: Total market value of the cards pulled from the pack.
        """
        if pack_id < 0 or pack_id >= self.num_packs:
            raise ValueError("Invalid pack_id.")

        pack = self.packs[pack_id]
        total_value = 0

        # Draw cards and calculate total value
        total_value += self.draw_cards(pack["common_list"], pack["common_probabilities"], 4)
        total_value += self.draw_cards(pack["uncommon_list"], pack["uncommon_probabilities"], 3)
        total_value += self.draw_cards(pack["foil_list"], pack["foil_probabilities"], 3)
        
        return total_value

    def draw_cards(self, card_list, probabilities, num_cards):
        """
        Simulate drawing cards from a list and calculate their total market value.
        
        Parameters:
        - card_list (list): List of cards to draw from.
        - probabilities (dict): Probabilities for each card.
        - num_cards (int): Number of cards to draw.
        
        Returns:
        - float: Total market value of the drawn cards.
        """
        cards = np.random.choice(card_list, size=num_cards, p=list(probabilities.values()))
        return sum(self.market_values[card] for card in cards)

class ThompsonSampling:
    def __init__(self, pack_pool, budget=180):
        """
        Initialize the Thompson Sampling algorithm for the PACKPOOL.

        Parameters:
        - pack_pool (PACKPOOL): The PACKPOOL instance with packs to be sampled.
        - budget (int): Number of total pack openings allowed.
        """
        self.pack_pool = pack_pool
        self.num_packs = pack_pool.num_packs
        self.budget = budget
        
        # Beta distribution parameters for each pack
        self.alpha = np.ones(self.num_packs)  # Success count
        self.beta = np.ones(self.num_packs)   # Failure count
        
        # Store results
        self.total_rewards = np.zeros(self.num_packs)  # Cumulative market value for each pack
        self.total_attempts = np.zeros(self.num_packs)  # Number of times each pack was opened

    def run(self):
        """
        Run the Thompson Sampling algorithm for the specified budget.
        """
        epsilon = 1e-6  # Small value to ensure beta parameters remain positive

        for _ in range(self.budget):
            # Sample from the beta distribution for each pack
            theta_hat = np.random.beta(self.alpha, self.beta)
            
            # Choose the pack with the highest sampled mean
            selected_pack = np.argmax(theta_hat)
            
            # Open the selected pack and calculate its value
            reward = self.pack_pool.open_pack(selected_pack)
            
            # Normalize reward to a [0, 1] scale for beta distribution updates
            normalized_reward = reward / max(self.pack_pool.market_values.values())
            normalized_reward = np.clip(normalized_reward, 0, 1)  # Ensure reward is within [0, 1]
            
            # Update reward totals and counts
            self.total_rewards[selected_pack] += reward
            self.total_attempts[selected_pack] += 1
            
            # Update the beta distribution parameters
            self.alpha[selected_pack] += normalized_reward + epsilon
            self.beta[selected_pack] += (1 - normalized_reward) + epsilon


    def compute_regret(self):
        """
        Compute the regret for the Thompson Sampling process.

        Returns:
        - regret (float): Total regret across the budgeted openings.
        """
        # Find the optimal pack's average reward
        optimal_pack_value = max(
            np.sum([self.pack_pool.market_values[card] for card in pack["common_list"]]) * 4 / len(pack["common_list"]) +
            np.sum([self.pack_pool.market_values[card] for card in pack["uncommon_list"]]) * 3 / len(pack["uncommon_list"]) +
            np.sum([self.pack_pool.market_values[card] for card in pack["foil_list"]]) * 3 / len(pack["foil_list"])
            for pack in self.pack_pool.packs
        )
        
        # Compute regret as the difference between optimal and obtained values
        total_obtained_value = sum(self.total_rewards)
        regret = optimal_pack_value * self.budget - total_obtained_value
        return regret

    def results(self):
        """
        Print the results of the Thompson Sampling process.
        """
        for i in range(self.num_packs):
            print(f"Pack {i}:")
            print(f"  Total value: {self.total_rewards[i]:.2f}")
            print(f"  Times opened: {int(self.total_attempts[i])}")
        print(f"Total regret: {self.compute_regret():.2f}")

# Initialize a PACKPOOL with 5 packs
pack_pool = PACKPOOL(num_packs=5)

# Run Thompson Sampling to minimize regret
thompson = ThompsonSampling(pack_pool, budget=180)
thompson.run()

# Display the results
thompson.results()