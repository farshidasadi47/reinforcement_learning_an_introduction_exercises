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

    def __init__(self, k=10, n_instances=1000) -> None:
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


class ConstantGreedyEstimatorParallel:
    """
    This class represents base class for a multi-armed bandit agent that
    is trained using epsilon-greedy action selection with constant step
    size.
    ----------
    Attributes:
        _k (int):
            The number of arms in the bandit problem.
        _n_instances (int):
            The number of runs to simulate in parallel.
        alpha (float):
            Constant step size for updating action value estimations.
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

    def __init__(
        self, k=10, n_instances=1000, alpha=0.1, eps=0.1, Q0=0.0
    ) -> None:
        """
        Initialize an EstimatorParallel instance.
        ----------
        Args:
            k (int, default=10):
                 The number of arms in the bandit problem.
            n_instances (int, default=2000):
                The number of instances to simulate in parallel.
            alpha (float, default=0.1):
                Constant step size for updating action value
                estimations.
            eps (float, default=0.1):
                The exploration parameter epsilon for epsilon-greedy
                action selection.
            Q0 (float, default=0.0):
                Initial action value estimates.
        """
        self._k = k
        self._n_instances = n_instances
        self._alpha = alpha
        self._eps = eps
        self._Q = np.ones((n_instances, k)) * Q0
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

    def simulate(self, n_steps=1000):
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
            # Update the bandit
            bandit.update()
        return average_reward


class UCBEstimatorParallel(ConstantGreedyEstimatorParallel):
    """
    This class represents a multi-armed bandit agent using the UCB
    (Upper Confidence Bound) action selection method.
    ----------
    Attributes:
        _c (float):
            Exploration parameter controlling confidence bounds.
        _N (numpy.ndarray):
            An array storing the number of times each action has been
            chosen.
        _t (int):
            The current step or time.
        (Other attributes inherited from EstimatorParallel)
    """

    def __init__(self, k=10, n_instances=1000, alpha=0.1, c=0.1):
        """
        Initialize a UCBEstimatorParaller instance.
        ----------
        Args:
            k (int, default=10):
                The number of arms in the bandit problem.
            n_instances (int, default=2000):
                The number of instances to simulate in parallel.
            alpha (float, default=0.1):
                Constant step size for updating action value
                estimations.
            c (float, default=0.1):
                Exploration parameter controlling the level of
                confidence levels in action selection..
        """
        super().__init__(k, n_instances, alpha)
        self._c = c
        self._N = np.zeros((self._n_instances, self._k), dtype=int)
        self._t = 0

    def select_action(self):
        """
        Select actions for all instances using the UCB action selection
        method.
        ----------
        Returns:
            numpy.ndarray, int:
            An array containing the indices of selected actions.
        """
        self._t += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            # Context manager is ued since initial values of _N is zero.
            Q = self._Q + self._c * np.sqrt(np.log(self._t)) / self._N
            a = np.argmax(Q, axis=1)
        self._N[self._index, a] += 1
        return a


class GradientEstimatorParallel(ConstantGreedyEstimatorParallel):
    """
    This class represents a multi-armed bandit agent using the
    gradient-based action selection method.
    ----------
    Attributes:
        _Rbar (numpy.ndarray, float):
            An array storing the mean reward for each instance.
        _NRbar (int):
            The total number of times rewards are updated.
        _H (numpy.ndarray, float):
            An array storing the preference values for each action and
            instance. These preferences are used in the action selection
            probabilities.
        (Other attributes inherited from EstimatorParallel)
    """

    def __init__(self, k=10, n_instances=1000, alpha=0.1):
        """
        Initialize a GradientEstimatorParallel instance.
        ----------
        Args:
            k (int, default=10):
                The number of arms in the bandit problem.
            n_instances (int, default=2000):
                The number of instances to simulate in parallel.
            alpha (float, default=0.1):
                Constant step size for updating action value
                estimations.
        """
        super().__init__(k, n_instances, alpha)
        self._Rbar = np.zeros(n_instances)
        self._NRbar = 0
        self._H = np.zeros((self._n_instances, self._k), dtype=float)

    def _update_Rbar(self, R):
        """
        Update the mean reward estimate across all instances.
        """
        self._NRbar += 1
        self._Rbar = self._Rbar + (R - self._Rbar) / self._NRbar

    def _softmax(self):
        """
        Compute the softmax probabilities based on preference values.
        """
        return np.exp(self._H) / np.exp(self._H).sum(axis=1, keepdims=True)

    def select_action(self):
        """
        Select actions for all instances based on highest probability.
        Note that action with highest preference value has highest
        probability.
        """
        return self._H.argmax(axis=1)

    def update_estimates(self, R, a):
        """
        Update the preference values using gradient ascent.
        ----------
        Args:
            R (numpy.ndarray, float):
                The array of received rewards for all instances.
            a (numpy.ndarray, int):
                An array of selected actions for all instances.
        """
        mask = np.zeros((self._n_instances, self._k), dtype=bool)
        mask[self._index, a] = True
        P = self._softmax()  # Probabilities.
        H = self._H
        # Update preference for the selected actions.
        H[mask] = H[mask] + self._alpha * (R - self._Rbar) * (1 - P[mask])
        # Update preference for the non-selected actions.
        mask = np.where(mask == False)
        H[mask] = (
            H[mask]
            - self._alpha * ((R - self._Rbar)[..., np.newaxis] * P)[mask]
        )
        self._update_Rbar(R)


def plot_param_study(parameter_reward_data, n_avg, file_name=None):
    """
    Plots a results of the parameter study showing the relationship
    between different parameter values and average rewards obtained
    over the last specified number of steps.
    ----------
    Parameters:
        parameter_reward_data : (dict)
            A dictionary containing label as key and parameter-reward
            data as values. The parameter-reward data should be a a 2D
            list or numpy.ndarray with two rows representing parameter
            and reward values.
        n_avg : (int)
            The number of steps to consider for averaging rewards.
        file_name : (str, default-None)
            The name of the file to save the plot as a PDF. If not
            provided, the plot is not saved.
    ----------
    Returns:
        None
    """
    colors = ["r", "b", "g", "c", "m", "k"]
    for i, (label, (param, reward)) in enumerate(
        parameter_reward_data.items()
    ):
        plt.plot(param, reward, c=colors[i], label=label, marker="*")
    plt.xlabel("Parameter value")
    plt.ylabel(f"Average Rewards Over Last {n_avg} Steps")
    plt.xscale("log")
    plt.legend()
    if file_name is not None:
        plt.savefig(file_name + ".pdf", bbox_inches="tight", pad_inches=0.1)
    #
    plt.show()


def test_parallel():
    """
    Perform a parameter study comparing different multi-armed bandit
    estimator methods.
    Used lower number of n_runs for the shorter runtime. The range of
    appropriate parameters may vary based on this value.
    ----------
    Parameters:
        None
    ----------
    Returns:
        None
    """
    k = 10
    eps = 0.1
    alpha = 0.1
    n_steps = 10000
    n_avg = n_steps // 2
    n_runs = 1000
    file_name = "exercise_2_11"
    # Set up parameter study data.
    labels = [
        r"$\epsilon$-greedy",
        r"UCB",
        r"Greedy with optimistic initial values, $\alpha=0.1$",
        r"Gradient bandit",
    ]
    parameters = [
        2 ** np.arange(-10, -1, dtype=float),
        2 ** np.arange(3, 9, dtype=float),
        2 ** np.arange(5, 17, dtype=float),
        2 ** np.arange(-15, 1, dtype=float),
    ]
    generators = [
        lambda eps: ConstantGreedyEstimatorParallel(k, n_runs, alpha, eps),
        lambda c: UCBEstimatorParallel(k, n_runs, alpha, c),
        lambda Q0: ConstantGreedyEstimatorParallel(
            k, n_runs, alpha, eps=0, Q0=Q0
        ),
        lambda alpha: GradientEstimatorParallel(k, n_runs, alpha),
    ]
    study_pairs = []
    for label, params, generator in zip(labels, parameters, generators):
        for param in params:
            study_pairs.append((label, param, generator))
    # Simulate
    parameter_raward_data = {label: [] for label in labels}
    for label, param, generator in tqdm(study_pairs, desc="Estimators"):
        estimator = generator(param)
        reward = estimator.simulate(n_steps)
        reward = reward[-n_avg:].mean()
        parameter_raward_data[label].append((param, reward))
    # Plot data.
    parameter_raward_data = {
        label: np.array(parameter_raward_data[label]).T for label in labels
    }
    plot_param_study(parameter_raward_data, n_avg, file_name)


########## test section ################################################
if __name__ == "__main__":
    test_parallel()
