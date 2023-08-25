# %%
########################################################################
# This file is the script for exercise 2.5 of the book
# Reinforcement Learning: An Introduction 2nd Ediiton.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        _q_star (numpy.ndarray, float):
            An array representing the true action values for all
            instances and arms.
        _index (numpy.ndarray, int):
            Auxiliary array used for numpy advanced indexing.
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
        self._q_star = np.random.rand(n_instances, k) * 1e-6
        self._index = np.arange(n_instances)

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
        return np.random.normal(self._q_star[self._index, a])

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


class EstimatorParallel:
    """
    This class represents base class for a multi-armed bandit agent that
    is trained using epsilon-greedy action selection.
    ----------
    Attributes:
        _k (int):
            The number of arms in the bandit problem.
        _n_instances (int):
            The number of runs to simulate in parallel.
        _eps (float):
            The exploration parameter epsilon for epsilon-greedy action
            selection.
        _Q (numpy.ndarray, float):
            An array storing the estimated action values.
        _index (numpy.ndarray, int):
            Auxiliary array used for numpy advanced indexing.
        _bandit:
            An instance of K_ParalletBandit.
    """

    def __init__(self, k=10, n_instances=2000, eps=0.1, Q0=None) -> None:
        """
        Initialize an EstimatorParallel instance.
        ----------
        Args:
            k (int, default=10):
                 The number of arms in the bandit problem.
            n_instances (int, default=2000):
                The number of instances to simulate in parallel.
            eps (float, default=0.1):
                The exploration parameter epsilon for epsilon-greedy
                action selection.
            Q0 (numpy.ndarray, default=None):
                Initial action value estimates. If None, all estimates
                are initialized to zero.
        """
        self._k = k
        self._n_instances = n_instances
        self._eps = eps
        if Q0 is None:
            self._Q = np.zeros((n_instances, k))
        else:
            self._Q = Q0
        self._index = np.arange(n_instances)
        self._bandit = K_BanditParallel(k, n_instances)

    def select_exploit_action(self):
        """
        Select the exploiting action for each run based on the estimated
        action values.
        ----------
        Returns:
            numpy.ndarray, int:
            An array containing the indices of selected actions.
        """
        a = self._Q.argmax(axis=1)
        return a

    def select_explore_action(self, n):
        """
        Select random exploring actions for a specified number of runs.

        Args:
            n (int): The number of runs to select explore actions for.
        ----------
        Returns:
            numpy.ndarray, int:
            An array containing the indices of selected actions.
        """
        a = np.random.randint(0, self._k, size=n)
        return a

    def select_action(self):
        """
        Select actions for all instances using epsilon-greedy action
        selection.
        ----------
        Returns:
            numpy.ndarray, int:
            An array containing the indices of selected actions.
        """
        probability = np.random.rand(self._n_instances)
        a = self.select_exploit_action()
        a[probability < self._eps] = self.select_explore_action(
            len(a[probability < self._eps])
        )
        return a

    def percent_optimal(self, a, a_star):
        """
        Calculate the percentage of optimal actions taken across all
        instances.
        ----------
        Args:
            a (numpy.ndarray, int):
                An array of selected actions for all instances.
            a_star (numpy.ndarray, int):
                An array of optimal actions for all instances.
        ----------
        Returns:
            float:
                The percentage of optimal actions taken.
        """
        return np.count_nonzero(a == a_star) / self._n_instances * 100

    # @abstractmethod
    def update_estimates(self, R, a):
        """
        Update the action value estimates.
        """
        raise NotImplementedError("This should not be called.")

    def simulate(self, n_steps=10000):
        """
        Simulate the multi-armed bandit process using the provided
        components.
        ----------
        Args:
            n_steps (int, default=10000):
                The number of simulation steps.
        ----------
        Returns:
            numpy.ndarray (int), numpy.ndarray (int):
            Arrays containing average rewards and optimal action
            percentages for selected actions for each step.
        """
        bandit = self._bandit
        average_reward = np.zeros((n_steps), dtype=float)
        optimal_percent = np.zeros((n_steps), dtype=float)
        for step in tqdm(range(n_steps), leave=False):
            # Get optimal actions.
            a_star = bandit.optimal()
            # Play game.
            a = self.select_action()
            R = bandit.play(a)
            # Update estimator.
            self.update_estimates(R, a)
            # Calculate the statistics.
            average_reward[step] = R.mean()
            optimal_percent[step] = self.percent_optimal(a, a_star)
            # Update the bandit
            bandit.update()
        return average_reward, optimal_percent


class AvgEstimatorParallel(EstimatorParallel):
    """
    This class represents a multi-armed bandit agent using
    epsilon-greedy action selection and sample averages reward
    estimation.
    ----------
    Attributes:
        _N (numpy.ndarray):
            An array storing the number of times each action has been
            chosen.
        (Other attributes inherited from EstimatorParallel)
    """

    def __init__(self, k=10, n_runs=2000, eps=0.1, Q0=None) -> None:
        """
        Initialize an AvgEstimatorParallel instance.
        ----------
        Args:
            The same as its parent class.
        """
        super().__init__(k, n_runs, eps, Q0)
        self._N = np.zeros((self._n_instances, self._k), dtype=int)

    def update_estimates(self, R, a):
        """
        Update the action value estimates using sample averages reward
        estimation.
        ----------
        Args:
            R (numpy.ndarray, float):
                The array of received rewards for all instances.
            a (numpy.ndarray, int):
                An array of selected actions for all instances.
        """
        Q = self._Q
        self._N[self._index, a] += 1
        Q[self._index, a] = (
            Q[self._index, a]
            + (R - Q[self._index, a]) / self._N[self._index, a]
        )


class CteEstimatorParallel(EstimatorParallel):
    """
    This class represents a multi-armed bandit agent using
    epsilon-greedy action selection and sample averages reward
    estimation.
    ----------
    Attributes:
        _alpha (float):
            Constant step size for updating action value estimations.
        (Other attributes inherited from EstimatorParallel)
    """

    def __init__(
        self, k=10, n_instances=2000, eps=0.1, alpha=0.1, Q0=None
    ) -> None:
        """
        Initialize an AvgEstimatorParallel instance.
        ----------
        Args:
            k (int, default=10):
                See parent class for description.
            n_instances (int, default=2000):
                See parent class for description.
            eps (float, default=0.0):
                See parent class for description.
            alpha (float, default=0.1):
                Constant step size for updating action value
                estimations.
            Q0 (numpy.ndarray, default=None):
                See parent class for description.
        """
        super().__init__(k, n_instances, eps, Q0)
        self._alpha = alpha

    def update_estimates(self, R, a):
        """
        Update the action value estimates using constant step size
        reward estimation.
        ----------
        Args:
            R (numpy.ndarray, float):
                The array of received rewards for all instances.
            a (numpy.ndarray, int):
                An array of selected actions for all instances.
        """
        Q = self._Q
        Q[self._index, a] = Q[self._index, a] + self._alpha * (
            R - Q[self._index, a]
        )


def plot_results(rewards, percents, names, file_name=None):
    """
    Plot and visualize the results of multi-armed bandit simulations.
    ----------
    Args:
        rewards (List[numpy.ndarray]):
            A list of arrays containing average rewards for different
            simulations.
        percents (List[numpy.ndarray]):
            A list of arrays containing optimal action percentages for
            different simulations.
        names (List[str]):
            A list of names or labels for each simulation.
        file_name (str, optional):
            The base name of the saved image files. If provided, two
            separate images with "_a.png" and "_b.png" suffixes will be
            saved using this base name.
    """
    colors = ["r", "b", "g", "c", "m", "k"]

    # Plot average rewards
    fig_a = plt.figure()
    plt.gcf()
    for i, (reward, name) in enumerate(zip(rewards, names)):
        plt.plot(reward, c=colors[i], label=name)
    plt.xlabel("Steps")
    plt.ylabel("Average Rewards")
    plt.legend()
    # Plot optimal action percentages
    fig_b = plt.figure()
    plt.gcf()
    for i, (percent, name) in enumerate(zip(percents, names)):
        plt.plot(percent, c=colors[i], label=name)
    plt.xlabel("Steps")
    plt.ylabel(r"% relative to optimal")
    plt.legend()
    plt.show()
    if file_name is not None:
        fig_a.savefig(
            file_name + "_a.pdf", bbox_inches="tight", pad_inches=0.1
        )
        fig_b.savefig(
            file_name + "_b.pdf", bbox_inches="tight", pad_inches=0.1
        )


########## test section ################################################
def test_parallel():
    k = 10
    eps = 0.1
    alpha = 0.1
    n_steps = 10000
    n_runs = 2000
    file_name = "exercise_2_5"
    #
    avg_estimator = AvgEstimatorParallel(k, n_runs, eps)
    cte_estimator = CteEstimatorParallel(k, n_runs, eps, alpha)
    #
    rewards = []
    percents = []
    estimators = [avg_estimator, cte_estimator]
    for estimator in tqdm(estimators, desc="Estimators"):
        reward, percent = estimator.simulate(n_steps)
        rewards.append(reward)
        percents.append(percent)
    # Plot results.
    plot_results(rewards, percents, ["average", "constant"], file_name)


########## test section ################################################
if __name__ == "__main__":
    test_parallel()
