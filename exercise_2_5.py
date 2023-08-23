# %%
########################################################################
# This file is the script for exercise 2.5 of the book
# Reinforcement Learning: An Introduction 2nd Ediiton.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1000)


########## Classes and Methods #########################################


class K_BanditParallel:
    """
    This class runs different instances of k-arm bandit in parallel.
    ----------
    Attributes:
        _k (int):
            The number of arms in the bandit problem.
        _n_instances (int):
            The number of bandit instances to run in parallel.
        _cte (numpy.ndarray, int):
            This arrays is used to facilitate indexing bandit instances.
        _q_star (numpy.ndarray, float):
            An array representing the true action values for all
            instances and arms.
    """

    def __init__(self, k=10, n_instances=2000) -> None:
        """
        Initialize a K_BanditParallel instance.
        ----------
        Args:
            k (int, default=10):
                The number of arms in the bandit problem.
            n_instances (int, default=2000):
                The number of bandit instances to run in parallel.
        """
        self._k = k
        self._n_instances = n_instances
        self._cte = np.arange(n_instances) * k
        self._q_star = np.random.rand(n_instances, k) * 1e-6

    def play(self, a):
        """
        Play the k-arm bandit game based on the given action.
        ----------
        Args:
            a (numpy.ndarray, int):
                The chosen actions. Length of array should be equal to
                number of instances (_n_instances).
        ----------
        Returns:
            numpy.ndarray, float:
                Array of rewards obtained from the chosen actions for
                instances.
        """
        return np.random.normal(self._q_star.flatten()[a + self._cte])

    def optimal(self):
        """
        Determine the optimal action for each bandit instance.
        ----------
        Returns:
            numpy.ndarray, float:
                An array containing the indices of optimal actions for
                each instance.
        """
        return self._q_star.argmax(axis=1)

    def update(self):
        """
        Update the action values using a random walk update.
        """
        self._q_star += np.random.normal(
            scale=0.01, size=(self._n_instances, self._k)
        )
