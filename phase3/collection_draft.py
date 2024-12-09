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

                "common_list": random.sample(cglobal_list, c_per_box),
                "uncommon_list": random.sample(uglobal_list, u_per_box),
                "foil_list": random.sample(fglobal_list, f_per_box),
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

        # Calculate expected values of packs (for regret analysis, but not "known" to agent)
        self.packEVs = []
        for pack in self.packs:
            packEV = 0
            packEV += sum(self.market_values[card] * pack["common_probabilities"][card] for card in pack["common_list"])
            packEV += sum(self.market_values[card] * pack["uncommon_probabilities"][card] for card in pack["uncommon_list"])
            packEV += sum(self.market_values[card] * pack["foil_probabilities"][card] for card in pack["foil_list"])
            self.packEVs.append(packEV)

    def cardListToDict(self, card_nums):
        """
        :param cards: len(allcards) list of integers giving number drawn of each card
        :return: dict containing same information but indexed by card name keys
        """
        card_keys = list(self.market_values.keys())
        return {card_keys[n]: card_nums[n] for n in range(len(card_keys))}

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

class ThompsonSampling:
    def __init__(self, packpool, num_trials=180):
        """
        Initialize Thompson Sampling with Dirichlet priors.
        
        Parameters:
        - packpool (PACKPOOL): Instance of the PACKPOOL class.
        - num_trials (int): Number of pack openings to simulate.
        """
        self.packpool = packpool
        self.num_trials = num_trials
        """
        self.alphas = np.ones((packpool.num_packs, len(packpool.packs[0]["common_list"]) +
                               len(packpool.packs[0]["uncommon_list"]) +
                               len(packpool.packs[0]["foil_list"])))
        """
        # Need priors on all the cards, not just the ones that happen to be in first pack:
        self.alphas = np.ones((packpool.num_packs, len(cglobal_list) + len(uglobal_list) + len(fglobal_list)))
        self.values_list = np.array(list(packpool.market_values.values())) # Ordered list of market values, converted from dict
        self.regret = []
        self.total_rewards = np.zeros(packpool.num_packs)
        self.times_opened = np.zeros(packpool.num_packs, dtype=int)

    def play_one_step(self):
        """
        Perform one step of Thompson Sampling:
        - Sample pack means from Dirichlet prior.
        - Open the pack with the highest sampled mean.
        - Update the Dirichlet prior based on the observed rewards.
        """
        sampled_means = []
        
        for pack_id in range(self.packpool.num_packs):
            # Sample from Dirichlet prior
            sampled_means.append(np.random.dirichlet(self.alphas[pack_id]))

        # Convert sampled means to total expected reward for each pack
        #expected_rewards = [sum(mean) for mean in sampled_means]
        expected_rewards = np.array([np.dot(mean, self.values_list) for mean in sampled_means])

        # Choose the pack with the highest expected reward
        #chosen_pack = np.argmax(expected_rewards)
        chosen_pack = random_argmax(expected_rewards)

        # Simulate opening the chosen pack
        #reward = self.packpool.open_pack(chosen_pack)
        # self.total_rewards[chosen_pack] += reward
        pack_cards = self.packpool.open_pack_list(chosen_pack)
        self.total_rewards[chosen_pack] += packpool.computeDeckValue(pack_cards)
        self.times_opened[chosen_pack] += 1

        # Update the posterior distribution
        # Normalize reward to fit within Dirichlet update range
        """
        normalized_reward = reward / max(self.packpool.market_values.values())
        self.alphas[chosen_pack] += normalized_reward
        """
        self.alphas[chosen_pack] += np.array(pack_cards)

        # Track regret
        """
        optimal_reward = max(
            self.packpool.open_pack(i) for i in range(self.packpool.num_packs)
        )
        self.regret.append(optimal_reward - reward)
        """
        self.regret.append(max(self.packpool.packEVs) - self.packpool.packEVs[chosen_pack])

    def run(self):
        """
        Execute Thompson Sampling for the specified number of trials.
        
        Returns:
        - cumulative_regret (np.ndarray): Cumulative regret over the trials.
        """
        for _ in range(self.num_trials):
            self.play_one_step()
        
        # Print final results for each pack
        print("\nFinal Results:")
        for pack_id in range(self.packpool.num_packs):
            print(f"Pack {pack_id + 1}:")
            print(f"  Total Value: {self.total_rewards[pack_id]:.2f}")
            print(f"  Times Opened: {self.times_opened[pack_id]}")
            print(f"  Average Value: {(self.total_rewards[pack_id]/self.times_opened[pack_id]):.2f}")

        return np.cumsum(self.regret)

class DefinedCollection:
    def __init__(self, packpool, raw_target, budget=1000, threshold=15, pack_cost=50):
        self.packpool = packpool
        self.target = {key: 0. for key in allcards}

        # Convert target list into a dictionary with counts for each card
        for card in raw_target:
            if card in self.target.keys():
                self.target[card] += 1
            else:
                print(f"Warning: {card} does not exist.")
        
        self.gauge = np.array(list(self.target.values()))

        # UCB-VI related values
        self.num_packs = packpool.num_packs
        self.pack_values = np.zeros(self.num_packs)  # Estimated values for each pack
        self.pack_counts = np.zeros(self.num_packs)  # Number of times each pack has been opened
        self.total_steps = 0  # Total steps taken

        # Dynamic budget adjustment
        self.initial_budget = budget  # Initial budget
        self.budget = budget  # Initial budget
        self.expanded_budget = 0  # Additional budget from high-value cards
        self.threshold = threshold  # Value threshold for high-value cards
        self.pack_cost = pack_cost # Cost of a single pack

    def single_reward(self, card):
        """
        Compute reward for a single card.
        - Full reward for cards in the target collection.
        - Partial reward for high-value cards above a threshold.
        """
        if self.target[card] > 0:
            return 1.0  # Full reward
        elif (self.packpool.market_values[card] > self.threshold) & (self.budget < self.initial_budget):
            return 0.1  # Partial reward
        elif (self.packpool.market_values[card] > self.threshold):
            return -0.2  # Negative reward
        else:
            return 0.0  # No reward

    def play_one_step(self):
        """
        Execute one step of the UCB-VI algorithm:
        - Select a pack based on the upper confidence bound.
        - Open the pack, draw cards, and update values and counts.
        """
        # Compute UCB for each pack
        if self.total_steps == 0:
            ucb_values = np.inf * np.ones(self.num_packs)
        else:
            ucb_values = (
                self.pack_values
                + np.sqrt(2 * np.log(self.total_steps + 1) / (self.pack_counts + 1))
            )
        
        # Select the pack with the highest UCB value
        pack_id = np.argmax(ucb_values)
        
        # Open the pack and draw cards
        drawn_cards = self.packpool.open_pack_list(pack_id)
        
        # Compute rewards
        reward = 0
        for card, count in zip(allcards, drawn_cards):
            card_value = self.packpool.market_values[card]
            reward += self.single_reward(card) * count
            
            # Expand budget for high-value cards above threshold
            if card_value > self.threshold:
                self.budget += card_value * count
                self.expanded_budget += card_value * count

        # Update pack values and counts
        self.pack_counts[pack_id] += 1
        step_size = 1 / self.pack_counts[pack_id]
        self.pack_values[pack_id] += step_size * (reward - self.pack_values[pack_id])
        
        # Update the total steps
        self.total_steps += 1
        
        # Update the target collection (reduce gauge for collected cards)
        self.gauge -= np.minimum(drawn_cards, self.gauge)
    
    def run(self):
        """
        Run the UCB-VI algorithm until the target collection is completed or the budget is exhausted.
        """
        while self.budget >= self.pack_cost:
            if np.all(self.gauge <= 0):
                print("Target collection completed!")
                break
            self.play_one_step()
            self.budget -= self.pack_cost
            print("step", self.total_steps, "complete")
            print("budget at ", self.budget, "expect", self.budget/self.pack_cost, "pulls")

        print("Final target status:", self.gauge)
        print("Pack values:", self.pack_values)
        print("Pack counts:", self.pack_counts)
        print("Total packs pulled:", self.total_steps)
        print("Expanded budget contributed:", self.expanded_budget)

# packpool = PACKPOOL(num_packs=5)
# ts = ThompsonSampling(packpool, num_trials=180)
# cumulative_regret = ts.run()

# target = input("Provide a list of desired cards: ")
target = "toad toad toadFoil turtle turtle turtleFoil firedragon firedragon firedragonFoil"

packpool = PACKPOOL(num_packs=9)
target = np.array(target.split())
collection = DefinedCollection(packpool, target, 1000)
collection.run()
