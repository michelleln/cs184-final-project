## TODO: relevant imports?
import numpy as np

# Used for tie-breaking (copied directly from HW2)
def random_argmax(a):
    """
    Select the index corresponding to the maximum in the input list.
    Ties are randomly broken.
    """
    return np.random.choice(np.where(a == a.max())[0])

# TODO: decide if this should really be implemented as a class, or something else?
class Thompson_sampling:
    # TODO: adapt MAB class attributes to new "Box" class, or whatever we use
    def __init__(self, Boxes):
        self.Boxes = Boxes
        self.K = Boxes.get_K() # Number of packs to choose from
        self.T = Boxes.get_T() # TODO: do we need a time horizon? TS doesn't explicitly depend on it
        self.J = Boxes.total_cards # TODO: implement getter for total number of cards to draw from
        self.cards_per_box = Boxes.cards_per_box # TODO: implement cards_per_box in Boxes class

        # Initialize with uniform priors for each packs and each card
        self.alphas = np.ones([self.K, self.J])

    def reset(self):
        """
        Reset the instance and eliminate history.
        """
        self.Boxes.reset()

    # TODO: implement function that computes value of cards drawn from (must choose values...)
    def reward_mapping(self, cards):
        pass

    def play_one_step(self):
        # Sample pack means from Dirichlet prior
        theta_hat = np.zeros(self.alpha.shape)
        r_hat = np.zeros(self.K)
        for box in range(self.K):
            theta_hat[box] = np.random.dirichlet(self.alphas[box], self.cards_per_box)
            r_hat[box] = self.reward_mapping(theta_hat[box])

        # Play arm with the highest sampled mean
        box_to_pull = random_argmax(r_hat)
        pulled_cards = self.Boxes.pull(box_to_pull) # TODO: change to using box class, or however we store this!
        reward = reward_mapping(pulled_cards)

        # Use drawn cards to calculate the posterior distribution
        self.alphas[arm] += pulled_cards
        return reward
