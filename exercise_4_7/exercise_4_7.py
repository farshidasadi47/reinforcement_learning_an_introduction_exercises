# %%
########################################################################
# This file is the script for exercise 4.7 of the book
# Reinforcement Learning: An Introduction 2nd Ediiton.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
from multiprocessing import Process
import numpy as np

np.set_printoptions(precision=1, suppress=True)
from scipy.special import factorial
import matplotlib.pyplot as plt
from pylab import cm

np.random.seed(1000)


########## Classes and Methods #########################################
def msg_header(msg):
    """
    This function takes a string and formats it into a header with
    asterisks on both sides.
    ----------
    Args:
        msg (str):
            The message to be included in the header.
    ----------
    Returns:
        msg (str):
            A formatted message header with asterisks.
    """
    N = len(msg)
    start_width = 10
    line_width = 79
    if N > 0:
        msg = " ".join(
            [
                "*" * start_width,
                msg,
                "*" * (line_width - (start_width + 2 + N)),
            ]
        )
    else:
        msg = "*" * line_width
    return msg


def poisson(lamb, n):
    """
    Calculate the Poisson probability mass function for given
    parameters.
    ----------
    Args:
        lamb (float):
            The average rate of events (lambda) over the given interval.
        n (numpy.ndarray, float):
            An array or list of non-negative integers representing the
            number of events to calculate the probability for.
    ----------
    Returns:
        numpy.ndarray, float
            An array of calculated Poisson probabilities corresponding
            to the values in 'n'.
    """
    n = np.asarray(n, dtype=float)
    return np.exp(-lamb) * (lamb**n) / factorial(n)


class CarRental:
    """
    A class representing the MDP for a car rental system. This is based
    on original car rental system presented in example 4.2 of the book.
    ----------
    Attributes:
        _reward_per_rent (int):
            The reward earned for renting a car.
        gamma (float):
            The discount factor for future rewards.
        max_car (int):
            The maximum number of cars that can be at each location.
        _cost_per_move (int):
            The cost incurred for relocating cars between locations.
        max_move (int):
            The maximum number of cars that can be relocated in a single
            night.
        _lamb_req_a (float):
            The Poisson parameter for the number of rental requests at
            location A.
        _lamb_req_b (float):
            The Poisson parameter for the number of rental requests at
            location B.
        _lamb_ret_a (float):
            The Poisson parameter for the number of car returns at
            location A.
        _lamb_ret_b (float):
            The Poisson parameter for the number of car returns at
            location B.
        states (list of tuples):
            A list of all possible states, where each state is a tuple
            representing the number of cars at locations A and B.
        transitions : dict
            A dictionary mapping (state, action) tuples to lists of
            (probability, next_state_A, next_state_B, reward) tuples.
    """

    def __init__(self) -> None:
        # Specifications of the system.
        self._reward_per_rent = 10
        self.gamma = 0.9
        self.max_move = 5
        self.max_car = 20
        self._cost_per_move = 2.0
        # Parameters related to request and return distribution.
        self._lamb_req_a = 3
        self._lamb_req_b = 4
        self._lamb_ret_a = 3
        self._lamb_ret_b = 2
        # Other parameters.
        self.states = [
            (i, j)
            for i in range(self.max_car + 1)
            for j in range(self.max_car + 1)
        ]
        # Get the MDP table.
        self.transitions = self._set_transitions()

    def get_action_bounds(self, state):
        """
        Get the lower and upper bounds for the number of cars that can
        be relocated based on the current state.
        ----------
        Args:
            state (tuple,int):
                A tuple representing the current state (number of cars
                at locations A and B).
        ----------
        Returns:
            lb_action, ub_action (tuple of int):
                The lower and upper bound for the number of cars that
                can be relocated.
        """
        state_a, state_b = state
        lb_action = max(-self.max_move, -state_b)
        ub_action = min(self.max_move, state_a)
        return lb_action, ub_action

    def _set_transitions(self):
        """
        Set transition probabilities for all possible state-action
        pairs.
        ----------
        Returns:
            transitions (dict):
                A dictionary mapping (state, action) tuples to lists of
                (probability, next_state_A, next_state_B, reward)
                tuples.
        """
        transitions = dict()
        for state in self.states:
            lb_action, ub_action = self.get_action_bounds(state)
            for action in range(lb_action, ub_action + 1):
                p_sp_r = self._get_transitions(state, action)
                transitions[(*state, action)] = p_sp_r
        return transitions

    def _get_relocating_reward(self, state_a, state_b, action):
        """
        Calculate the reward for relocating cars between locations.
        ----------
        Args:
            state_a (int):
                The number of cars at location A after the relocation.
            state_b (int):
                The number of cars at location B after the relocation.
            action (int):
                The number of cars to relocate from A to B (negative for
                relocating from B to A).
        ----------
        Returns:
            relocating_reward (float):
                The reward obtained from relocating cars.
        """
        return -self._cost_per_move * np.abs(action)  # Cost per move.

    def _get_center_transition(self, state_i, lamb_req_i, lamb_ret_i):
        """
        Calculate ( p_i(s',r|s), s'_i, r_i ) for a single location.
        ----------
        Args:
            state_i (int):
                The number of cars at the current location.
            lamb_req_i (float):
                The Poisson parameter for car requests at the location.
            lamb_ret_i (float):
                The Poisson parameter for car returns at the location.
        ----------
        Returns:
            p (numpy.ndarray, float):
                An array of transition probabilities to next_states.
            next_states (numpy.ndarray, int):
                An array of next states for the location.
            r (numpy.ndarray, float)"
                An array of rewards associated with transitioning to
                next_states.
        """
        # Build p(requested, returned) for all combinations.
        p_req = poisson(lamb_req_i, np.arange(state_i))
        # Add probability of requests >= current number of cars.
        p_req = np.append(p_req, 1 - sum(p_req))
        p_ret = poisson(lamb_ret_i, np.arange(self.max_car))
        # Add probability of returning >= max cars.
        p_ret = np.append(p_ret, 1 - sum(p_ret))
        prob = np.meshgrid(p_ret, p_req)
        prob = prob[0] * prob[1]  # p(requested, returned)
        # Build r(requesed, returned).
        rets, reqs = np.meshgrid(
            np.arange(self.max_car + 1),
            np.arange(state_i + 1),
        )
        rewards = self._reward_per_rent * np.arange(state_i + 1)
        # Build next state as next[reqs, rets].
        s_p = np.clip(state_i - reqs + rets, 0, self.max_car)
        #
        p = np.zeros(self.max_car + 1, dtype=float)
        r = np.zeros(self.max_car + 1, dtype=float)
        for next_state in np.arange(s_p.max() + 1):
            p[next_state] = np.sum(prob[np.where(s_p == next_state)])
            reward_weight = np.where(s_p == next_state, prob, 0).sum(axis=1)
            r[next_state] = np.average(rewards, weights=reward_weight)
        next_states = np.arange(self.max_car + 1, dtype=int)
        return p, next_states, r

    def _combine_centers_transitions(self, p_sp_r_centers, r0):
        """
        Combines dynamics of different centers to the overall dynamics
        ( p(s',r|s), s', r ).
        ----------
        Args:
        p_sp_r_centers (tuple of tuples):
            A tuple of tuples, where each inner tuple is transition
            probabilities, next_states_a, next_states_a, and rewards for
            a specific center.
        r0 (int):
            The immediate reward obtained from relocating cars between
            locations.
        ----------
        Returns:
            p (numpy.ndarray, float):
                An array of combined transition probabilities to
                next_states.
            next_states_a (numpy.ndarray, int):
                An array of combined next states for location a.
            next_states_b (numpy.ndarray, int):
                An array of combined next states for location b.
            r (numpy.ndarray, float)
                An array of combined rewards associated with
                transitioning to next_states.
        """
        # Combine centers.
        (p_a, next_states_a, r_a), (p_b, next_states_b, r_b) = p_sp_r_centers
        # p[next_state_a, next_state_b].
        p = np.meshgrid(p_b, p_a)
        p = p[0] * p[1]
        # next_states.
        next_states_b, next_states_a = np.meshgrid(
            next_states_b, next_states_a
        )
        # r[next_state_a, next_state_b].
        r_b, r_a = np.meshgrid(r_b, r_a)
        r = r_a + r_b + r0
        #
        p = p.ravel()
        next_states_a = next_states_a.ravel()
        next_states_b = next_states_b.ravel()
        r = r.ravel()
        return p, next_states_a, next_states_b, r

    def _get_transitions(self, state, action):
        """
        This function returns all possible (s', r) pairs with their
        probability as list of tuples ( p(s',r|s,a), s'_a, s'_b, r ).
        ----------
        Args:
        state (tuple of int):
            A tuple representing the current state (number of cars at
            locations A and B).
        action (int):
            The number of cars to relocate from A to B (negative for
            relocating from B to A).
        ----------
        Returns:
        p_sp_r (tuple of numpy.ndarrays):
            A tuple containing ( p(s',r|s,a), s'_a, s'_b, r ).
        """
        state_a, state_b = state
        # Apply action.
        state_a = int(state_a - action)
        state_b = int(state_b + action)
        # Calculate immediate cost.
        r0 = self._get_relocating_reward(state_a, state_b, action)
        # Calculate each center's transition dynamics.
        centers = [
            (state_a, self._lamb_req_a, self._lamb_ret_a),
            (state_b, self._lamb_req_b, self._lamb_ret_b),
        ]
        p_sp_r_centers = []
        for center in centers:
            state_i, lamb_req_i, lamb_ret_i = center
            p_sp_r_i = self._get_center_transition(
                state_i, lamb_req_i, lamb_ret_i
            )
            p_sp_r_centers += [p_sp_r_i]
        # Combine center transitions.
        p_sp_r = self._combine_centers_transitions(p_sp_r_centers, r0)
        return p_sp_r


class CarRentalModified(CarRental):
    """
    A class representing the MDP for a car rental system. This is based
    on modified car rental system presented in exercise 4.7 of the book.
    It inherits from CarRental class.
    ----------
    Attributes:
        _parking_limit (int):
            The initial parking limit of centers.
        _extra_parking_cost (int):
            The cost of having additional parking in each center.
        (Other attributes inherited from CarRental)
    """

    def __init__(self):
        self._parking_limit = 10
        self._extra_parking_cost = 4
        super().__init__()

    def _get_relocating_reward(self, state_a, state_b, action):
        """
        Calculate the reward for relocating cars between locations.
        ----------
        Args:
            state_a (int):
                The number of cars at location A after the relocation.
            state_b (int):
                The number of cars at location B after the relocation.
            action (int):
                The number of cars to relocate from A to B (negative for
                relocating from B to A).
        ----------
        Returns:
            relocating_reward (float):
                The reward obtained from relocating cars.
        """
        # Employee can move one car from a to b for free.
        if action > 0:
            action -= 1
        # Cost per move.
        relocating_reward = -self._cost_per_move * np.abs(action)
        # Parking fee.
        n_parking = n_parking = (1 if state_a > self._parking_limit else 0) + (
            1 if state_b > self._parking_limit else 0
        )
        relocating_reward -= self._extra_parking_cost * n_parking
        return relocating_reward


class PolicyIteration:
    """
    A class implementing the policy iteration with iterative policy
    evaluation for solving CarRental problem.
    ----------
    Attributes:
        theta (float):
            A small positive threshold value used to check for
            convergence.
        mdp (MarkovDecisionProcess, CarRental):
            An instance of the CarRental class.
        gamma (float):
            The discount factor for future rewards.
        states (list of tuples):
            A list of all possible states, where each state is a tuple
            representing the number of cars at locations a and b.
        transitions (dict):
            A dictionary mapping (state, action) tuples to lists of
            ( p(s',r|s,a), s'_a, s'_b, r ) tuples.
        V (list of numpy.ndarray, float):
            List of state value fuction tables at different iterations.
        pi (list of numpy.ndarray, int):
            A list of policy tables at different iterations.
    """

    def __init__(self, mdp) -> None:
        """
        Initialize a PolicyIteration instance.
        ----------
        Args:
            mdp : MarkovDecisionProcess
                An instance of the CarRental MDP process.
        """
        self.theta = 0.01
        self.mdp = mdp
        self.gamma = mdp.gamma
        self.states = mdp.states
        self.transitions = mdp.transitions
        # State value and policy tables.
        self.V = [np.zeros((mdp.max_car + 1, mdp.max_car + 1), dtype=float)]
        self.pi = [np.zeros((mdp.max_car + 1, mdp.max_car + 1), dtype=int)]

    def policy_evaluation(self):
        """
        Performs iterative policy evaluation based on the latest policy
        table.
        ----------
        Returns:
            None
        """
        V = self.V[-1].copy()  # Use as initial value.
        pi = self.pi[-1]  # Policy table.
        # Update V until convergence.
        i = 1
        while True:
            V_old = V.copy()
            delta = 0.0
            # Update V one iteration.
            for state in self.states:
                transition_key = tuple([*state, pi[state]])
                p, next_state_a, next_state_b, r = self.transitions[
                    transition_key
                ]
                V[state] = sum(
                    p * (r + self.gamma * V[next_state_a, next_state_b])
                )
            # Check criteria for convergence.
            delta = max(delta, np.abs(V_old - V).max())
            if delta < self.theta:
                break
            i += 1
        self.V += [V]
        print(f"\tIterative policy evaluation converged in {i} iterations.")

    def policy_improvement(self):
        """
        Improves the current policy based on the estimated state values.
        ----------
        Returns:
            policy_stable : bool
                A boolean value indicating whether the policy has become
                stable after the improvement. It returns True if the
                policy remains the same after the update, indicating
                convergence.
        """
        V = self.V[-1]
        pi_old = self.pi[-1]
        pi = np.zeros_like(pi_old)
        policy_stable = False
        # Update pi for all states.
        for state in self.states:
            lb_action, ub_action = self.mdp.get_action_bounds(state)
            # Try all possible inputs to find maximal action.
            actions = np.arange(lb_action, ub_action + 1, dtype=int)
            value = np.zeros_like(actions, dtype=float)
            for i, action in enumerate(actions):
                transition_key = tuple([*state, action])
                p, next_state_a, next_state_b, r = self.transitions[
                    transition_key
                ]
                value[i] = sum(
                    p * (r + self.gamma * V[next_state_a, next_state_b])
                )
            # Choose maximum, take first if there are multiple maximas.
            pi[state] = actions[value.argmax()]
        self.pi += [pi]
        # Check stopping criteria.
        if np.all(pi == pi_old):
            policy_stable = True
        return policy_stable

    def policy_iteration(self):
        """
        Perform the policy iteration algorithm to find the optimal
        policy.
        ----------
        Returns:
            V (list of numpy.nd.array, float):
                List of state value tables at consequtive iterations.
            pi (list of numpy.ndarray, int):
                List of policies at concequtive iterations.
        """
        i = 1
        while True:
            print(f"Policy iteraiton step {i}.")
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            if policy_stable:
                break
            i += 1
        print(f"Policy iteration is converged in {i} steps.")
        self.V.pop()
        self.pi.pop()
        return self.V, self.pi


def plot_policies(policies, state_values, title, file_name=None):
    """
    Plot policies and state values.
    ----------
    Parameters:
    -----------
        policies (list of numpy.ndarray, int):
            A list of policy tables.
        state_values (list of numpy.ndarray, float):
            A list of state values corresponding to policies.
        title (str):
            The title of the plot.
        file_name (str or None, optional):
            If provided, the plot will be saved with the given filename.
    ----------
    Returns:
        None
    """
    # Draw policies.
    policies = np.asarray(policies, dtype=int)
    state_value = np.asarray(state_values[-1], dtype=float)
    max_move = policies.max()
    min_move = policies.min()
    max_car = state_value.shape[-1] - 1
    n_plots = len(policies) + 1
    n_cols = 3
    n_rows = n_plots // n_cols + (1 if n_plots % n_cols else 0)
    fig, ax = plt.subplots(n_rows, n_cols, constrained_layout=True)
    fig.set_size_inches(8, 0.75 * 8.0)
    fig.suptitle(title)
    for i, policy in enumerate(policies):
        row = i // n_cols
        col = i % n_cols
        cmap = plt.get_cmap("jet", max_move - min_move + 1)
        plt.sca(ax[row, col])
        plt.imshow(
            policy,
            origin="lower",
            interpolation="none",
            vmin=min_move,
            vmax=max_move + 1,
            cmap=cmap,
        )
        plt.title(f"$\\pi_{i}$")
    plt.sca(ax[row, 0])
    plt.xlabel("# Cars at 2nd location")
    plt.ylabel("# Cars at 1st location")
    # Remove unused axes.
    for c in range(col + 1, n_cols):
        fig.delaxes(ax[row, c])
    # Add color bar.
    cbar = plt.colorbar(
        ax=ax[0, n_cols - 1], ticks=np.arange(min_move, max_move + 1) + 0.5
    )
    cbar = cbar.ax.set_yticklabels(np.arange(min_move, max_move + 1))
    # Plot state value table.
    X = np.arange(0, max_car + 1)
    Y = np.arange(0, max_car + 1)
    Y, X = np.meshgrid(Y, X)
    ax_new = fig.add_subplot(int(f"{n_rows}{n_cols}{i+2}"), projection="3d")
    surf = ax_new.plot_surface(Y, X, state_value, cmap=cm.jet)
    ax_new.grid(False)
    # Set axes limits.
    ax_new.set_xlim(0, max_car)
    ax_new.set_ylim(0, max_car)
    ax_new.set_zlim(0.99 * state_value.min(), 1.01 * state_value.max())
    # Set axes ticks.
    ax_new.set_xticks([0, max_car])
    ax_new.set_yticks([0, max_car])
    ax_new.set_zticks([state_value.min(), state_value.max()])
    # Set axes labels.
    ax_new.set_xlabel("# Cars at 2nd location")
    ax_new.set_ylabel("# Cars at 1st location")
    ax_new.set_zlabel("Value")
    # Add title.
    plt.sca(ax_new)
    plt.title(f"$v_{{\\pi_{i}}}$")
    # Save figure if requested.
    if file_name is not None:
        fig.savefig(file_name + ".pdf", bbox_inches="tight", pad_inches=0.1)


def example_4_2():
    """
    Reproduces the example 4.2.
    """
    print(msg_header("Example 4.2"))
    mdp = CarRental()
    policy_iteration = PolicyIteration(mdp)
    state_values, policies = policy_iteration.policy_iteration()
    title = "example_4_2"
    file_name = "example_4_2"
    plot_policies(policies, state_values, title, file_name)


def exercise_4_7():
    """
    Code for exercise 4.7.
    """
    print(msg_header("Exercise 4.7"))
    mdp = CarRentalModified()
    policy_iteration = PolicyIteration(mdp)
    state_values, policies = policy_iteration.policy_iteration()
    title = "exercise 4.7"
    file_name = "exercise_4_7"
    plot_policies(policies, state_values, title, file_name)


########## test section ################################################
if __name__ == "__main__":
    example_4_2()
    exercise_4_7()
    plt.show()
