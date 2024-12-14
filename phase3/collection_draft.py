import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import beta

random.seed(184)
np.random.seed(184) # Apparently this also seeds scipy.stats

cglobal_list = ["mushroom", "moth", "bramble", "mantis", "beetle", "pollenpuff", "crown", "goat", "fox", "horse", "magma", "anteater", "lizard", "centipede", "duck", 
                "jellyfish", "seal", "clam", "crab", "seahorse", "goldfish", "swan", "cucumber", "triggerfish", "snowmoth", "ball", "electric", "jolt", "zebra", 
                "frill", "urchin", "fairy", "slow", "mime", "model", "stamp", "golem", "pangolin", "mole", "ape", "boomerang", "kick", "punch", "drill", "martial", 
                "octopus", "snake", "bat", "maw", "blades", "rat", "sparrow", "cat", "leek", "trio", "tongue", "egg", "bull", "data", "chinchilla", "sheep"]
uglobal_list = ["toad", "butterfly", "bee", "flower", "pitcher", "tree", "firedragon", "dog", "flare", "firebird", "turtle", "frog", "starfish", "carp", "nessie", 
                "vapor", "nautilus", "icebird", "ninja", "mouse", "magnet", "zapbird", "eel", "spoons", "ghost", "hypnosis", "experiment", "lady", "wrestler", 
                "boulder", "rocksnake", "horseshoe", "queen", "king", "sludge", "gas", "boltnut", "dragon", "pidgeon", "song", "parent", "mimic", "pterodactyl",
                "bear"]
fglobal_list = ["toadFoil", "treeFoil", "firedragonFoil", "dogFoil", "firebirdFoil", "turtleFoil", "starfishFoil", "icebirdFoil", "mouseFoil", "zapbirdFoil", "ghostFoil", "experimentFoil", "wrestlerFoil", "boomerangFoil",
                "songFoil", "originFoil"] # Must have distinct card names as these are used in dict keys for e.g. market values

allcards = cglobal_list + uglobal_list + fglobal_list

c_per_box = 18 #38 #14
u_per_box = 8 #14 # 4
f_per_box = 3 #6 # 2

# Used for tie-breaking
def random_argmax(a):
    """
    Select the index corresponding to the maximum in the input list.
    Ties are randomly broken.
    """
    return np.random.choice(np.where(a == a.max())[0])

class PACKPOOL:
    def __init__(self, num_packs=5, pack_init="random", initial_budget=1000):
        """
        Initialize the PACKPOOL with a fixed number of packs and set market values for cards.
        
        Parameters:
        - num_packs (int): Number of packs in the pool.
        - pack_init (str): Method to initialize pack distributions
        - initial_budget (float): Starting budget for buying packs/cards
        """
        # Store initial budget
        self.initial_budget = initial_budget
        self.budget = initial_budget
        
        # Initialize collection tracking
        self.collection = np.zeros(len(allcards))  # Current collection status
        self.target = {card: 0 for card in allcards}  # Target collection
        self.gauge = np.zeros(len(allcards))  # Cards still needed
        
        # Initialize pack tracking
        self.num_packs = num_packs
        self.pack_data = np.zeros([num_packs, len(allcards)])
        self.alphas = np.ones([num_packs, len(allcards)])
        
        # Define fixed market values for all cards
        self.market_values = self.set_market_values()
        
        # Pack cost
        self.pack_cost = max(5 * c_per_box + 10 * u_per_box + 20 * f_per_box, 100)

        # Initialize packs completely randomly
        if pack_init == "random":
            self.packs = [
                {
                    "common_list": random.sample(cglobal_list, c_per_box),
                    "uncommon_list": random.sample(uglobal_list, u_per_box),
                    "foil_list": random.sample(fglobal_list, f_per_box),
                    "common_probabilities": None,
                    "uncommon_probabilities": None,
                    "foil_probabilities": None
                }
                for _ in range(self.num_packs)
            ]
        
        # Initialize packs by looping through card lists
        elif pack_init == "loop":
            self.packs = []
            for i in range(self.num_packs):
                pack = {}

                # Common list
                start_idx = i % len(cglobal_list)
                end_idx = (i * c_per_box) % len(cglobal_list)
                if start_idx < end_idx:
                    pack["common_list"] = cglobal_list[start_idx:end_idx]
                else:
                    pack["common_list"] = cglobal_list[start_idx:] + cglobal_list[:end_idx]

                # Uncommon list
                start_idx = i % len(uglobal_list)
                end_idx = (i * c_per_box) % len(uglobal_list)
                if start_idx < end_idx:
                    pack["uncommon_list"] = uglobal_list[start_idx:end_idx]
                else:
                    pack["uncommon_list"] = uglobal_list[start_idx:] + uglobal_list[:end_idx]

                # Foil list
                start_idx = i % len(fglobal_list)
                end_idx = (i * f_per_box) % len(fglobal_list)
                if start_idx < end_idx:
                    pack["foil_list"] = fglobal_list[start_idx:end_idx]
                else:
                    pack["foil_list"] = fglobal_list[start_idx:] + fglobal_list[:end_idx]

                # Default probabilities
                pack["common_probabilities"] = None
                pack["uncommon_probabilities"] = None
                pack["foil_probabilities"] = None

                self.packs.append(pack)

        # Assign probabilities for each pack
        for pack in self.packs:
            pack["common_probabilities"] = self.assign_probabilities(pack["common_list"])
            pack["uncommon_probabilities"] = self.assign_probabilities(pack["uncommon_list"])
            pack["foil_probabilities"] = self.assign_probabilities(pack["foil_list"])

        # Calculate expected values of packs
        self.packEVs = []
        for pack in self.packs:
            packEV = 0
            packEV += sum(self.market_values[card] * pack["common_probabilities"][card] for card in pack["common_list"])
            packEV += sum(self.market_values[card] * pack["uncommon_probabilities"][card] for card in pack["uncommon_list"])
            packEV += sum(self.market_values[card] * pack["foil_probabilities"][card] for card in pack["foil_list"])
            self.packEVs.append(packEV)
            
    def reset(self):
        """Reset the environment to initial state"""
        # Reset collections and gauge with float type
        self.collection = np.zeros(len(allcards), dtype=np.float32)
        self.gauge = np.array(list(self.target.values()), dtype=np.float32)
        
        # Reset budget and other tracking
        self.budget = float(self.initial_budget)
        self.pack_data = np.zeros([self.num_packs, len(allcards)], dtype=np.float32)
        self.alphas = np.ones([self.num_packs, len(allcards)], dtype=np.float32)
        
        return self.get_state()

    def is_collection_complete(self):
        """Check if target collection is complete (all gauge values <= 0)"""
        return np.all(self.gauge <= 0)

    def set_target_collection(self, target_cards):
        """Set the target collection to aim for"""
        self.target = {card: 0 for card in allcards}  # Reset target
        for card in target_cards:
            if card in self.target:
                self.target[card] += 1
            else:
                print(f"Warning: {card} does not exist.")
        # Convert gauge to float type
        self.gauge = np.array(list(self.target.values()), dtype=np.float32)
        # Initialize collection as float type
        self.collection = np.zeros_like(self.gauge, dtype=np.float32)

    def get_state_space_size(self):
        """Return the size of the state space"""
        return len(self.collection) + 1  # Collection status + budget

    def get_action_space_size(self):
        """Return size of action space"""
        return (self.num_packs + len(allcards) * 2)  # Buy/sell actions for each card

    def get_all_card_names(self):
        """Return ordered list of all card names"""
        return allcards

    def get_card_values(self):
        """Return ordered list of card values matching allcards order"""
        return [self.market_values[card] for card in allcards]

    def get_state(self):
        """Return current state representation"""
        state = np.concatenate([
            self.collection,  # Current collection status
            [self.budget]     # Current budget
        ])
        # Ensure state is float32
        return state.astype(np.float32)

    def step(self, action):
        """
        Execute action and return results.
        
        Args:
            action: int, index of action to take
        
        Returns:
            next_state: np.array of new state
            reward: float
            done: bool
        """
        if action < self.num_packs:  # Buy pack
            if self.budget < self.pack_cost:
                return self.get_state(), -1.0, True  # Penalize invalid action
                
            drawn_cards = self.open_pack_list(action)
            # Ensure drawn_cards is float type
            drawn_cards = drawn_cards.astype(np.float32)
            cost = self.pack_cost
            
            self.budget -= cost
            self.collection += drawn_cards
            self.gauge -= np.minimum(drawn_cards, self.gauge)
            
            # Calculate reward
            collection_reward = np.sum(np.minimum(drawn_cards, self.gauge))
            reward = collection_reward * 10.0 - cost * 0.1  # Scale rewards
            
        else:
            card_idx = (action - self.num_packs) // 2
            is_sell = (action - self.num_packs) % 2 == 0
            card_value = list(self.market_values.values())[card_idx]

            if is_sell:  # Sell card
                if self.collection[card_idx] <= 0:
                    return self.get_state(), -1.0, True  # Penalize invalid action
                    
                self.collection[card_idx] -= 1
                self.budget += card_value
                reward = card_value if self.gauge[card_idx] <= 0 else -1.0
                
            else:  # Buy card
                if self.budget < card_value:
                    return self.get_state(), -1.0, True  # Penalize invalid action
                    
                self.collection[card_idx] += 1
                self.budget -= card_value
                self.gauge[card_idx] = max(0, self.gauge[card_idx] - 1)
                reward = 10.0 if self.gauge[card_idx] == 0 else -card_value

        done = self.is_collection_complete() or self.budget < self.pack_cost
        return self.get_state(), reward, done


    def cardListToDict(self, card_nums):
        """
        :param cards: len(allcards) list of integers giving number drawn of each card
        :return: dict containing same information but indexed by card name keys
        """
        card_keys = list(self.market_values.keys())
        return {card_keys[n]: card_nums[n] for n in range(len(card_keys))}

    def fillOutDict(self, partial_dict):
        """
        Parameters:
            partial_dict: dict with cards as key values, with possibly not all cards represented, e.g. as in pack["common_list"]

        Returns:
            complete_dict: dict with the same card numbers as partial_dict, but with all cards represented as keys
        """
        full_dict = {key: 0 for key in allcards}
        for card in list(partial_dict.keys()):
            full_dict[card] = partial_dict[card]

        return full_dict


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

    def open_pack_list(self, pack_id):
        """
        Simulate opening a pack and return the cards drawn

        Parameters:
        - pack_id (int): Index of the pack to open.

        Returns:
        - float: Total market value of the cards pulled from the pack.
        """
        if pack_id < 0 or pack_id >= self.num_packs:
            raise ValueError("Invalid pack_id.")

        pack = self.packs[pack_id]
        drawn_cards = np.zeros(len(allcards))

        # Draw cards and compile cards_list
        #print(self.draw_cards_list(pack["common_list"], pack["common_probabilities"], 4)) # TEST
        drawn_cards += self.draw_cards_list(pack["common_list"], pack["common_probabilities"], 4)
        drawn_cards += self.draw_cards_list(pack["uncommon_list"], pack["uncommon_probabilities"], 3)
        drawn_cards += self.draw_cards_list(pack["foil_list"], pack["foil_probabilities"], 3)

        return drawn_cards

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

    def draw_cards_list(self, card_list, probabilities, num_cards):
        """
        Simulate drawing cards from a list

        Parameters:
        - card_list (list): List of cards to draw from.
        - probabilities (dict): Probabilities for each card.
        - num_cards (int): Number of cards to draw.

        Returns:
        - int: NumPy array with number of cards drawn
        """
        cards = np.random.choice(card_list, size=num_cards, p=list(probabilities.values())) # Returns list of dict keys
        cards_dict = {key: 0 for key in allcards}
        for card in cards:
            cards_dict[card] += 1

        return np.array(list(cards_dict.values()))

    def computeDeckValue(self, deck):
        """
        :param deck: list of integers describing number of each card drawn
        :return: value of drawn cards
        """
        return np.dot(np.array(list(packpool.market_values.values())), np.array(deck)) # TODO: are np.array's really necessary?

class DefinedCollection:
    def __init__(self, packpool, raw_target, budget=1000, threshold=15, pack_cost=50):
        """
        packpool: PACKPOOL class instance
        raw_target: string, desired cards written out separated by spaces
        budget: int, budget for purchasing card packs
        threshold: int, value above which a card is deemed worthy of selling
        pack_cost: int, cost for each card pack opened
        """
        self.packpool = packpool
        self.target = {key: 0. for key in allcards}

        # Convert target list into a dictionary with counts for each card
        for card in raw_target:
            if card in self.target.keys():
                self.target[card] += 1
            else:
                print(f"Warning: {card} does not exist.")

        # UCB related values
        self.c = 0.1 # Hyperparameter describing how much weight to give UCB reward bonus
        self.delta = 0.95 # Hyperparameter describing confidence level for UCB
        self.num_packs = packpool.num_packs
        self.pack_values = np.zeros(self.num_packs)  # Estimated values for each pack
        self.pack_counts = np.zeros(self.num_packs)  # Number of times each pack has been opened
        self.total_steps = 0  # Total steps taken

        self.gauge = np.array(list(self.target.values()))  # List of cards still needed to be collected
        self.collection = np.zeros_like(self.gauge)  # Cards collected but not sold
        self.pack_data = np.zeros([self.num_packs, len(allcards)]) # All cards collected so far, indexed by pack number
        self.alphas = np.ones([self.num_packs, len(allcards)]) # Dirichlet priors on the pack distributions


        # Dynamic budget adjustment
        self.initial_budget = budget  # Initial budget
        self.budget = budget  # Initial budget
        self.expanded_budget = 0  # Additional budget from high-value cards
        self.threshold = threshold  # Value threshold for high-value cards
        self.pack_cost = pack_cost # Cost of a single pack

        # Add reward scaling to match PPO
        self.collection_reward_scale = 10.0
        self.budget_reward_scale = 0.5


    def compute_reward(self, old_collection, new_collection, budget_change):
        """Compute reward similar to PPO for comparison"""
        collection_progress = np.sum(
            np.minimum(new_collection, self.gauge) - 
            np.minimum(old_collection, self.gauge)
        )
        collection_reward = collection_progress * self.collection_reward_scale
        
        budget_reward = budget_change if budget_change > 0 else budget_change * self.budget_reward_scale
        
        return collection_reward + budget_reward
    
    
    def play_one_step(self):
        """
        Execute one step of the algorithm:
        - Select a pack based on the upper confidence bound.
        - Open the pack, draw cards, and update values and counts.
        """
        # Store initial state for reward calculation
        old_collection = self.collection.copy()
        old_budget = self.budget

        # Estimate "baseline" reward for each pack
        card_rewards = np.zeros(len(allcards))
        for i, card in enumerate(allcards):
            card_rewards[i] = self.single_reward(card)

        pack_rewards = np.zeros(self.num_packs)
        card_means = np.zeros([self.num_packs, len(allcards)])
        for i in range(self.num_packs):
            # Expectation value of multinomial with Dirichlet prior
            card_means[i] = (c_per_box + u_per_box + f_per_box) * self.alphas[i] / np.sum(self.alphas[i])
            pack_rewards[i] = np.dot(card_rewards, card_means[i])

        # Select the pack with the highest expected reward
        pack_id = random_argmax(pack_rewards)

        # Open the pack and draw cards
        drawn_cards = self.packpool.open_pack_list(pack_id)

        # Update data and priors on drawn pack
        self.pack_data[pack_id] += drawn_cards
        self.alphas[pack_id] += drawn_cards
        
        # Process drawn cards and handle selling
        for i, (card, count) in enumerate(zip(allcards, drawn_cards)):
            card_value = self.packpool.market_values[card]
            # Sell cards if they're valuable and not needed
            if card_value > self.threshold and self.gauge[i] == 0:
                self.budget += card_value * count
                drawn_cards[i] -= count
                self.expanded_budget += card_value * count

        # Update collection and gauge
        self.collection += drawn_cards
        self.gauge -= np.minimum(drawn_cards, self.gauge)

        # Update pack statistics
        self.pack_counts[pack_id] += 1
        self.total_steps += 1

        # Calculate final reward
        budget_change = self.budget - old_budget
        reward = self.compute_reward(old_collection, self.collection, budget_change)

        return reward
    
    def run(self):
        """
        Run the state-dependent reward bonus algorithm until the target collection is completed or the budget is exhausted.
        """
        while self.budget >= self.pack_cost:
            if np.all(self.gauge <= 0):
                print("Target collection completed!")
                break
            self.play_one_step()
            self.budget -= self.pack_cost
            print("step", self.total_steps, "complete")
            print("budget at ", self.budget, "expect", self.budget/self.pack_cost, "pulls")

        print("\nFinal Results:")
        print(f"Target status: {self.gauge}")
        print(f"Collection status: {self.collection}")
        print(f"Pack values: {self.pack_values}")
        print(f"Pack counts: {self.pack_counts}")
        print(f"Total packs pulled: {self.total_steps}")
        print(f"Cards missing: {sum(self.gauge)}")
        print(f"Expanded budget: {self.expanded_budget}")
        
        # Plot final distribution comparison
        self.plot_pack_distributions()
        
        # Plot learning progress
        self.plot_distribution_distance()
        
        # Calculate final KL divergence
        final_kl = self.pack_distribution_distance()
        print("\nFinal KL divergence for each pack:")
        for i, kl in enumerate(final_kl):
            print(f"Pack {i+1}: {kl:.4f}")

    # TODO: write function to calculate our how accurate our knowledge of the pack distributions is (use KL-divergence)
    def pack_distribution_distance(self):
        """Calculate KL divergence between true and learned pack distributions"""
        kl_divs = np.zeros(self.num_packs)
        
        for i in range(self.num_packs):
            # Get learned distribution
            learned_dist = self.pack_data[i]
            if self.pack_counts[i] != 0:
                learned_dist = learned_dist / np.sum(learned_dist)
                
            # Get true distribution
            pack = self.packpool.packs[i]
            true_probs = {**pack["common_probabilities"], 
                        **pack["uncommon_probabilities"], 
                        **pack["foil_probabilities"]}
            true_dist = np.array(list(self.packpool.fillOutDict(true_probs).values()))
            
            # Calculate KL divergence, avoiding divide by zero
            nonzero_mask = (true_dist != 0) & (learned_dist != 0)
            if np.any(nonzero_mask):
                kl_div = np.sum(true_dist[nonzero_mask] * 
                            np.log(true_dist[nonzero_mask] / learned_dist[nonzero_mask]))
                kl_divs[i] = kl_div
                
        return kl_divs


    def plot_pack_distributions(self):
        fig, axes = plt.subplots(2, self.num_packs, figsize=(20, 4))
        axes = axes.ravel()  # Flatten the 2D array into a 1D array for iteration

        # Plot the experimentally-determined pack distributions
        expt_probs = np.zeros_like(self.pack_data)
        for i in range(np.shape(self.pack_data)[0]):
            # Exception if pack was never pulled (zero data)
            if self.pack_counts[i] != 0:
                expt_probs[i] = self.pack_data[i] / np.sum(self.pack_data[i])
            else:
                expt_probs[i] = self.pack_data[i]
        ymax = np.max(expt_probs)

        #for i, (ax, data) in enumerate(zip(axes, self.pack_data)):
        for i, (ax, data) in enumerate(zip(axes, expt_probs)):
            packDict = self.packpool.cardListToDict(data)

            keys = list(packDict.keys())
            values = list(packDict.values())
            ax.bar(keys, values, width=0.8, edgecolor="black")
            ax.set_title(f"Experiment, Pack {i + 1}")
            ax.set_xlabel("Cards")
            ax.set_xticks([])
            ax.set_ylim(bottom=0, top=ymax)
            ax.set_ylabel("Probabilities")

        # Plot the actual pack distributions below
        ymax = 0
        for pack in self.packpool.packs:
            pack_probs = {**pack["common_probabilities"], **pack["uncommon_probabilities"], **pack["foil_probabilities"]}
            ymax = max(ymax, max(list(pack_probs.values())))

        for i, (ax, pack) in enumerate(zip(axes[self.num_packs:], self.packpool.packs)):
            pack_probs = {**pack["common_probabilities"], **pack["uncommon_probabilities"], **pack["foil_probabilities"]}
            full_pack_probs = self.packpool.fillOutDict(pack_probs)

            keys = list(packDict.keys())
            values = list(full_pack_probs.values())

            ax.bar(keys, values, width=0.8, edgecolor="black")
            ax.set_title(f"Actual, Pack {i + 1}")
            ax.set_xlabel("Cards")
            ax.set_xticks([])
            ax.set_ylim(bottom=0, top=ymax)
            ax.set_ylabel("Probabilities")

        plt.show()
        pass

    # TODO: plot the KL-divergence as a function of packs opened
    def plot_distribution_distance(self):
        """Plot KL divergence over time for each pack"""
        # Calculate KL divergence for each timestep
        kl_history = []
        pack_data_cumsum = np.zeros_like(self.pack_data)
        
        for t in range(1, self.total_steps + 1):
            # Update running sum of pack distributions
            pack_data_cumsum += self.pack_data
            
            # Calculate KL divergence at this timestep
            kl_divs = np.zeros(self.num_packs)
            for i in range(self.num_packs):
                if self.pack_counts[i] > 0:
                    # Get learned distribution up to this point
                    learned_dist = pack_data_cumsum[i] / np.sum(pack_data_cumsum[i])
                    
                    # Get true distribution
                    pack = self.packpool.packs[i]
                    true_probs = {**pack["common_probabilities"], 
                                **pack["uncommon_probabilities"], 
                                **pack["foil_probabilities"]}
                    true_dist = np.array(list(self.packpool.fillOutDict(true_probs).values()))
                    
                    # Calculate KL divergence
                    nonzero_mask = (true_dist != 0) & (learned_dist != 0)
                    if np.any(nonzero_mask):
                        kl_div = np.sum(true_dist[nonzero_mask] * 
                                    np.log(true_dist[nonzero_mask] / learned_dist[nonzero_mask]))
                        kl_divs[i] = kl_div
                        
            kl_history.append(kl_divs)

        # Plot KL divergence over time
        plt.figure(figsize=(10, 5))
        kl_history = np.array(kl_history)
        for i in range(self.num_packs):
            plt.plot(kl_history[:, i], label=f'Pack {i+1}')
        plt.xlabel('Steps')
        plt.ylabel('KL Divergence')
        plt.title('Pack Distribution Learning Progress')
        plt.legend()
        plt.yscale('log')  # Log scale for better visualization
        plt.grid(True)
        plt.show()

def calculate_collection_stats(collection, target_collection):
    """
    Calculate collection statistics.
    
    Args:
        collection: np.array of current collection
        target_collection: dict of target numbers for each card
    
    Returns:
        dict of statistics
    """
    target_array = np.array([target_collection.get(card, 0) for card in allcards])
    missing = np.maximum(0, target_array - collection)
    
    return {
        'total_collected': np.sum(collection),
        'unique_collected': np.sum(collection > 0),
        'missing_cards': np.sum(missing),
        'completion_percentage': (1 - np.sum(missing) / np.sum(target_array)) * 100
    }
if __name__ == "__main__":
    # TEST CASE 1: Basic Collection
    print("\nTEST CASE 1: Basic Collection")
    print("-" * 50)
    packpool = PACKPOOL(num_packs=5, pack_init="loop")
    target = "mushroom mushroom toadFoil turtle turtle turtleFoil firedragon firedragon originFoil"
    target = np.array(target.split())
    maxPackValue = 5 * c_per_box + 10 * u_per_box + 20 * f_per_box
    collection = DefinedCollection(packpool, target, budget=5000, pack_cost=maxPackValue)
    collection.run()
    # KL divergence plots will be shown automatically from the modified run() method

    """
    # TEST CASE 2: Single Card Collection
    print("\nTEST CASE 2: Single Card Collection")
    print("-" * 50)
    packpool = PACKPOOL(num_packs=5, pack_init="loop")
    target = "originFoil " * 10  # Want 10 copies
    target = np.array(target.split())
    maxPackValue = 5 * c_per_box + 10 * u_per_box + 20 * f_per_box
    collection = DefinedCollection(packpool, target, budget=10000, pack_cost=maxPackValue)
    collection.run()

    # TEST CASE 3: Completionist Collection
    print("\nTEST CASE 3: Completionist Collection")
    print("-" * 50)
    packpool = PACKPOOL(num_packs=5, pack_init="loop")
    # Create target with one of each card
    target = " ".join(cglobal_list + uglobal_list + fglobal_list)
    target = np.array(target.split())
    maxPackValue = 5 * c_per_box + 10 * u_per_box + 20 * f_per_box
    collection = DefinedCollection(packpool, target, budget=5000, pack_cost=maxPackValue)
    collection.run()
    """
