# %%
########################################################################
# This file is the script for exercise 4.9 of the book
# Reinforcement Learning: An Introduction 2nd Ediiton.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import numpy as np

np.set_printoptions(precision=1, suppress=True)
import matplotlib.pyplot as plt
from pylab import cm

np.random.seed(1000)


########## Classes and Methods #########################################
class ValueIteration:
    """
    Value Iteration is a class for solving a simplified gambling problem
    using the value iteration algorithm.
    ----------
    Args:
        p_h (float):
            The probability of winning a gamble.
        s_max (int, default=100, optional)
            The maximum amount of money the gambler can have.
        theta (float, default=1e-6, optional):
            A threshold for convergence.
    ----------
    Attributes:
        p_h (float):
            The probability of winning a gamble.
        s_max (int):
            The maximum amount of money the gambler can have.
        theta (float):
            A threshold for convergence.
        s (numpy.ndarray, int):
            An array of possible states (amounts of money) from 1 to
            s_max-1.
        pi (list of numpy array of int):
            A list to store the optimal policy for each state.
        V (list of numpy arrays of float):
            A list to store the value function for each iteration.
    ----------
    Methods:
        _value_iteration():
            Performs one step of the value iteration algorithm.
        value_iteration():
            Executes entire value iteration process until convergence.
        plot(file_name=None):
            Plots optimal value and policy.
    """

    def __init__(self, p_h, s_max=100, theta=1e-6) -> None:
        self.p_h = p_h
        self.s_max = s_max
        self.theta = theta
        self.s = np.arange(1, s_max)
        self.pi = []
        self.V = [np.zeros(s_max + 1, dtype=float)]
        self.V[0][-1] = 1

    def _value_iteration(self):
        """
        Perform one step of the value iteration algorithm.
        ----------
        Returns:
            bool: True if the iteration has converged, False otherwise.
        """
        converged = False
        V_old = self.V[-1]
        V = V_old.copy()
        pi = [0.0]  # Optimal policy when s = 0.
        # Go over all states.
        for s in self.s:
            # Possible actions and consequences.
            a = np.arange(1, min(s, self.s_max - s) + 1, dtype=int)
            s_lose = s - a
            s_win = s + a
            # value= p(s',r|s,a)(r+ V_k(s))
            values = self.p_h * V[s_win] + (1 - self.p_h) * V[s_lose]
            V[s] = values.max()
            pi += [a[values.argmax()]]
        self.V += [V]
        pi = pi + [0]  # Add optimal policy when s = s_max.
        self.pi += [np.array(pi, dtype=int)]
        if abs(V_old - V).max() < self.theta:
            converged = True
        return converged

    def value_iteration(self):
        """
        Executes the entire value iteration process until convergence.
        """
        i = 1
        while True:
            print(f"Value iteration step {i}.")
            converged = self._value_iteration()
            if converged:
                break
            i += 1
        print("=" * 72 + f"\nValue iteration converged in {i} steps.")

    def plot(self, file_name=None):
        """
        Plots optimal state value and policy table.
        ----------
        Args:
            file_name (str, default=None, optional):
                If provided, the plot will be saved with the given
                filename.
        """
        fig_value = plt.figure()
        s = np.arange(self.s_max + 2) - 0.5
        indexes = [1, 2, 3, len(self.V) // 2, -1]
        colors = ["b", "r", "g", "darkorange", "k"]
        for idx, i in enumerate(indexes):
            label = f"sweep {i}" if i > 0 else "final sweep"
            plt.stairs(self.V[i], s, color=colors[idx], label=label)
        plt.xlim([0, self.s_max])
        plt.legend(loc="lower right")
        plt.title(f"Value estimeate for $p_h=$ {self.p_h}")
        plt.ylabel("Value estimates")
        plt.xlabel("Capital")
        if file_name is not None:
            fig_value.savefig(file_name + f"_value_ph_{self.p_h}.pdf")
        #
        fig_policy = plt.figure()
        plt.stairs(self.pi[-1], s)
        plt.xlim([0, self.s_max])
        plt.title(f"Actions for $p_h=$ {self.p_h}")
        plt.ylabel("Final policy (stake)")
        plt.xlabel("Capital")
        if file_name is not None:
            fig_policy.savefig(file_name + f"_action_ph_{self.p_h}.pdf")


def exercise_4_9():
    """
    Code for exercise 4.9.
    """
    file_name = "exercise_4_9"
    for p_h in [0.25, 0.4, 0.55]:
        state_value = ValueIteration(p_h)
        state_value.value_iteration()
        state_value.plot(file_name)
    plt.show()


########## test section ################################################
if __name__ == "__main__":
    exercise_4_9()
