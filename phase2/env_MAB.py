import numpy as np
import random

class MAB:
    def __init__(self, T=100, mu_list=[0.3,0.5,0.6,0.65,0.7]):
        '''
        Parameters:
            T: horizon
            mu_list: list of true values for bandits
            seed: random seed to make grading easier
        '''
        self.__K = len(mu_list)
        self.__mu_list = mu_list
        self.__T = T
        self.__record = np.zeros((self.__K,2))
        self.__regrets = []

    def pull(self,ind):
        '''
        Pull the bandit with index ind
        '''
        reward = 1 * (random.random() < self.__mu_list[ind])
        self.__record[ind,reward] += 1
        self.__regrets.append(max(self.__mu_list) - self.__mu_list[ind])
        return reward
    
    def reset(self):
        '''
        Reset the bandit
        '''
        self.__record = np.zeros((self.__K,2))
        self.__regrets = []

    def get_record(self):
        '''
        Get the current record
        '''
        return self.__record
    
    def get_regrets(self):
        '''
        '''
        return np.cumsum(self.__regrets)

    def get_T(self):
        '''
        '''
        return self.__T

    def get_K(self):
        '''
        '''
        return self.__K        
