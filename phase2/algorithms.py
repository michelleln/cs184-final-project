from env_MAB import *


def random_argmax(a):
    """
    Select the index corresponding to the maximum in the input list.
    Ties are randomly broken.
    """
    return np.random.choice(np.where(a == a.max())[0])


class Explore:
    def __init__(self, MAB):
        self.MAB = MAB
        self.K = MAB.get_K()
        self.T = MAB.get_T()

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        record = self.MAB.get_record()
        num_pulls = np.sum(record, axis=1)

        # Use given function for tie-breaking
        arm = random_argmax(-1 * num_pulls) # -1 to convert to argmin
        self.MAB.pull(arm)

        """
        # Sort arms by number of times pulled
        sorted_pulls = np.argsort(num_pulls)

        # Break ties
        least_pulled = [sorted_pulls[0]]
        i = 1
        next_least_pulled = sorted_pulls[i]
        while next_least_pulled == least_pulled:
            sorted_pulls.append(next_least_pulled)
            i += 1
            next_least_pulled = sorted_pulls[i]

        k = len(least_pulled)
        arm_ind = np.random.random_integers(0, k) # Draw pull uniformly from least-pulled arms
        arm = least_pulled[arm_ind]

        # Pull the arm and record the reward
        self.MAB.pull(arm)
        """


class Greedy:
    def __init__(self, MAB):
        self.MAB = MAB
        self.K = MAB.get_K()
        self.T = MAB.get_T()
        self.greedy_arms = []

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        record = self.MAB.get_record()
        num_pulls_tot = int(np.sum(record))  # Use fact that this is a Bernoulli bandit, so sum(record) is the # of pulls

        # Exploration phase:
        if num_pulls_tot < self.K:
            self.MAB.pull(num_pulls_tot) # Since zero-indexed, should work out to pull 1st unpulled arm

        # Save the greedy arms after K pulls
        elif num_pulls_tot == self.K:
            self.greedy_arms = np.where(record[:, 1] == 1)[0] # indices of arms that succeeded in first round

            # Handle exception where no arms were successful in first round:
            if len(self.greedy_arms) == 0:
                self.greedy_arms = list(range(self.K))

            arm = np.random.choice(self.greedy_arms)
            self.MAB.pull(arm)

        # Greedy phase:
        else:
            arm = np.random.choice(self.greedy_arms)
            self.MAB.pull(arm)


class ETC:
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        self.delta = delta
        self.K = MAB.get_K()
        self.T = MAB.get_T()
        self.best_arms = []

        # Use formula given in assignment to determine length of exploration phase:
        self.Ne = np.floor( (self.T * np.sqrt( np.log(2 * self.K / self.delta) / 2 ) / self.K)**(2/3) )

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        record = self.MAB.get_record()
        num_pulls = (np.sum(record, axis=1)).astype(int)

        # Exploration phase:
        if np.sum(num_pulls) < (self.Ne * self.K):
            # TODO: this make sense?
            arm = random_argmax(-1 * num_pulls) # -1 to convert argmax to argmin
            self.MAB.pull(arm)

        # Save the arms with the best empirical means after the exploration phase:
        elif np.sum(num_pulls) == (self.Ne * self.K):
            best_mean = np.max(record[:, 1])
            self.best_arms = np.where(record[:, 1] == best_mean)[0]  # indices of arms that succeeded in first round

            arm = np.random.choice(self.best_arms)
            self.MAB.pull(arm)

        # Exploitation phase:
        else:
            arm = np.random.choice(self.best_arms)
            self.MAB.pull(arm)

class Epgreedy:
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        self.K = MAB.get_K()
        self.T = MAB.get_T()

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        record = self.MAB.get_record()
        t = int(np.sum(record))

        # Exploration phase:
        if t < self.K:
            self.MAB.pull(t)  # Since zero-indexed, should work out to pull 1st unpulled arm

        # Exploitation phase:
        else:
            eps_t = ( self.K * np.log(t) / t )**(1/3)
            p_rand = min(eps_t, 1)

            # Explore:
            if np.random.random() < p_rand:
                arm = np.random.randint(low=0, high=self.K) # Completely random arm
                self.MAB.pull(arm)

            # Exploit:
            else:
                # Now must actively monitor empirical arm means
                arm_means = record[:, 1] / (record[:, 0] + record[:, 1]) # All arms pulled at least once, so no div-by-0
                """
                best_mean = np.max(arm_means)
                best_arms = np.where(arm_means == best_mean)[0]
                arm = np.random.choice(best_arms)
                """
                arm = random_argmax(arm_means)
                self.MAB.pull(arm)

class UCB:
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        self.K = MAB.get_K()
        self.T = MAB.get_T()
        self.delta = delta

    def reset(self):
        """
        Reset the instance and eliminate history.
        """
        self.MAB.reset()

    def play_one_step(self):
        record = self.MAB.get_record()

        # Compute empirical reward estimate
        rewards_k = record[:, 1]
        pulls_k = record[:, 0] + record[:, 1]
        mu_tk = rewards_k / pulls_k # Gives np.nan for divide-by-0 (will handle later)

        # Compute the upper confidence bound
        upper_bounds = mu_tk + np.sqrt( np.log(self.K * self.T / self.delta) / pulls_k )
        upper_bounds[pulls_k == 0] = np.inf # TODO: does this work for the argmax?

        arm = random_argmax(upper_bounds)
        self.MAB.pull(arm)


class Thompson_sampling:
    def __init__(self, MAB):
        self.MAB = MAB
        self.K = MAB.get_K()
        self.T = MAB.get_T()

        # Initialize with uniform priors for each arm
        self.alpha = np.ones(self.K)
        self.beta  = np.ones(self.K)

    def reset(self):
        """
        Reset the instance and eliminate history.
        """
        self.MAB.reset()

    def play_one_step(self):
        # Sample arm means from beta prior
        theta_hat = np.random.beta(self.alpha, self.beta) # TODO: check that array args work correctly?

        # Play arm with the highest sampled mean
        arm = random_argmax(theta_hat)
        reward = self.MAB.pull(arm)

        # Use reward to calculate the posterior distribution
        self.alpha[arm] += reward
        self.beta[arm]  += 1 - reward
