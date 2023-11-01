# %%
########################################################################
# This file is the script for exercise 5.12 of the book
# Reinforcement Learning: An Introduction 2nd Ediiton.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
# Sections of the code are inspired by Git repositories listed below:
# 1: github.com/terrence-ou/Reinforcement-Learning-2nd-Edition-Notes-Codes
# 2: github.com/vojtamolda/reinforcement-learning-an-introduction
########## Libraries ###################################################
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

np.set_printoptions(precision=1, suppress=True)
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


class Track:
    """
    Class for creating racetrack maps.
    -----------
    Attributes:
        _map (numpy.ndarray):
            A 2D NumPy array representing the racetrack map.
    -----------
    Methods:
        track_a(cls):
            Creates and returns a Track instance with the track A map.
        track_b(cls):
            Creates and returns a Track instance with the track B map.
    """

    """
    Class for creating racetrack environment for reinforcement learning.
    -----------
    Attributes:
        _map (numpy.ndarray):
            A 2D NumPy array representing the racetrack map.
        _window_size (tuple):
            A tuple representing the size of the racetrack window
            (rows, columns).
        noise (bool):
            A boolean flag indicating whether noise should be applied to
            actions.
        _start_poses (tuple):
            A tuple containing the row and column indices of the starting
            positions on the map.
        _max_speed (int):
            The maximum speed a vehicle can have.
        _min_speed (int):
            The minimum speed a vehicle can have.
        _actions (dict):
            A dictionary mapping action indices to corresponding 2D
            accelerations.
        _state (numpy.ndarray):
            A 1D NumPy array representing the current state of the vehicle.
        _previous_state (numpy.ndarray):
            A 1D NumPy array representing the previous state of the vehicle.
        _fig_ax (tuple):
            A tuple containing the Matplotlib figure and axis used for
            rendering.
    -----------
    Methods:
        reset():
            Resets the vehicle's state to a random starting position.
        step(action):
            Takes a step in the environment by applying the specified action
            and returns the next state, reward, and terminal status.
        render():
            Renders the racetrack environment, including the vehicle's
            trajectory.
        generate_episode(policy, summary=False, render=False, title=None,
                         file_name=None):
            Generates an episode of interactions with the environment following
            a given policy and returns episode-related information.
    -----------
    Properties:
        map:
            Returns the racetrack map as a 2D NumPy array.
        n_action:
            Returns the number of available actions.
        n_state:
            Returns the size of the racetrack window (rows, columns).
        state:
            Returns the current state of the vehicle as a list.
    -----------
    Class Methods:
        track_a():
            Create a Track instance with the track A map.
        track_b():
            Create a Track instance with the track B map.
    """

    CELL_TYPES = {
        "gravel": 0,
        "start": 1,
        "track": 2,
        "finish": 3,
    }
    CMAP = ListedColormap(["darkgrey", "lightcoral", "white", "lightgreen"])

    # Cell types
    def __init__(self, track_map, noise=False):
        """
        Class constructore method.
        -----------
        Args:
            track_map (numpy.ndarray, int):
                A 2D NumPy array representing the racetrack map.
            noise (bool, optional):
                A boolean flag to apply noise to actions (default is False).
        """
        self._map = track_map[::-1, :]
        self._window_size = track_map.shape
        self._max_speed = 4
        self._min_speed = 0
        n_speeds = self._max_speed - self._min_speed + 1
        self._n_state = np.hstack((self._window_size, [n_speeds, n_speeds]))
        self.noise = noise
        self._start_poses = np.where(self._map == self.CELL_TYPES["start"])
        # Calculate padded map. Used for finish line crossing detection.
        self._padded_map = np.pad(
            self._map,
            (
                (abs(self._min_speed), abs(self._max_speed)),
                (abs(self._min_speed), abs(self._max_speed)),
            ),
            constant_values=self.CELL_TYPES["gravel"],
        )
        #
        self._actions = {
            0: np.array([-1, -1]),
            1: np.array([-1, 0]),
            2: np.array([-1, 1]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),
            5: np.array([0, 1]),
            6: np.array([1, -1]),
            7: np.array([1, 0]),
            8: np.array([1, 1]),
        }
        self._state = None
        self._previous_state = None
        self._fig_ax = None
        #
        self.reset()

    def _discretize_path(self, next_state, state):
        """
        Discretize the path between two states.
        -----------
        Args:
            next_state (numpy.ndarray, int):
                The next state.
            state (numpy.ndarray, int):
                The current state.
        -----------
        Returns:
            numpy.ndarray, int:
                2D array representing the discretized path between the states.
        """
        next_pos = next_state[:2]
        pos = state[:2]
        steps = np.round(np.linalg.norm(next_pos - pos)).astype(int)
        path = np.linspace(pos, next_pos, steps + 1).round().astype(int).T
        return path

    def _check_path_status(self, next_state, state):
        """
        Check the collision/finish status of the path between two states.
        -----------
        Args:
            next_state (numpy.ndarray, int):
                The next state.
            state (numpy.ndarray, int):
                The current state.
        -----------
        Returns:
            tuple:
                A tuple containing three boolean values indicating whether the
                path is within limits, on gravel, crossing finish line.
        """
        # Get path.
        path = self._discretize_path(next_state, state)
        # Chechk is path is within limits.
        clipped_path = np.clip(path, 0, self._n_state[:2, np.newaxis] - 1)
        within_limits = np.array_equal(path, clipped_path)
        # Get path values.
        path_values = self._map[clipped_path[0], clipped_path[1]]
        # Checke if path crosses grovel.
        on_gravel = ((path_values == self.CELL_TYPES["gravel"])).any()
        # Check if finish line is crossed.
        path_values = self._padded_map[path[0], path[1]]
        finish_crossed = (path_values == self.CELL_TYPES["finish"]).any()
        return within_limits, on_gravel, finish_crossed

    def reset(self):
        """
        Reset the vehicle's state to a random starting position.
        """
        start_poses = self._start_poses
        # Choose random startong position.
        rnd_idx = np.random.randint(start_poses[0].size)
        rnd_start_pos = [start_poses[0][rnd_idx], start_poses[1][rnd_idx]]
        # Build state.
        self._state = np.array(rnd_start_pos + [0, 0], dtype=int)

    def step(self, action):
        """
        Take a step in the environment by applying the specified action.
        -----------
        Args:
            action (int):
                The action index to be applied.
        -----------
        Returns:
            tuple:
                A tuple containing the next state, reward, and a boolean
                indicating if the state is terminal (corresponding path
                crossing finish line).
        """
        position = self._state[:2]
        velocity = self._state[2:]
        # Calculate acceleration.
        acceleration = self._actions.get(action, (0, 0))
        if np.random.rand() < 0.1 and self.noise:
            acceleration = np.array([0, 0])
        # Calculate next_velocity.
        next_velocity = np.clip(
            velocity + acceleration, self._min_speed, self._max_speed
        )
        if np.all(next_velocity == 0):
            next_velocity = velocity
        # Calculate next position.
        next_position = position + next_velocity
        # Chech path status.
        within_limits, on_gravel, finish_crossed = self._check_path_status(
            next_position, position
        )
        next_position = np.clip(next_position, 0, self._n_state[:2] - 1)
        state_is_terminal = finish_crossed
        # Build next state.
        self._previous_state = self._state
        self._state = np.hstack((next_position, next_velocity))
        # Determine if state is terminal.
        if on_gravel or (not within_limits and not finish_crossed):
            state_is_terminal = False
            self.reset()
        #
        return self.state, -1, state_is_terminal

    def _render_map(self):
        """
        Renders the racetrack map.
        -----------
        Returns:
            tuple:
                A tuple containing the Matplotlib figure and axis used for
                rendering.
        """
        fig, ax = plt.subplots()
        ax.imshow(self._map, aspect="equal", origin="lower", cmap=self.CMAP)
        # Add axes ticks.
        ax.set_xticks(
            np.arange(0, self._window_size[1], self._max_speed), minor=False
        )
        ax.set_yticks(
            np.arange(0, self._window_size[0], self._max_speed), minor=False
        )
        # Add grid lines
        ax.set_xticks(np.arange(0.5, self._window_size[1], 1), minor=True)
        ax.set_yticks(np.arange(0.5, self._window_size[0], 1), minor=True)
        ax.tick_params(which="minor", length=0)
        ax.grid(which="minor", color="dimgray", linestyle="-", linewidth=0.5)
        #
        return fig, ax

    def render(self):
        """
        Render the racetrack environment, including the vehicle's trajectory.
        -----------
        Returns:
            tuple:
                A tuple containing the Matplotlib figure and axis used for
                rendering.
        """
        # Draw map.
        if self._fig_ax is None:
            self._fig_ax = self._render_map()
        ax = self._fig_ax[1]
        # Draw robots.
        ax.annotate(
            "",
            xy=np.flip(self._state[:2]),
            xytext=np.flip(self._previous_state[:2]),
            arrowprops=dict(
                arrowstyle="-|>", color="black", lw=2, shrinkA=0, shrinkB=0
            ),
        )
        return self._fig_ax

    def _print_episode_summary(
        self, states, actions, p_actions, rewards, next_states, last100=True
    ):
        """
        Print a summary of the episode's interactions.
        -----------
        Args:
            states (list):
                List of states in the episode.
            actions (list):
                List of actions taken in the episode.
            p_actions (list):
                List of action probabilities in the episode.
            rewards (list):
                List of rewards received in the episode.
            next_states (list):
                List of next states in the episode.
            last100 (bool, optional, default=True):
                Flag to only print the last 100 steps of the episode.
        """
        base_step = max(0, len(rewards) - 100)
        print(msg_header("Episode summary"))
        for step, reward in enumerate(rewards[base_step:]):
            step += base_step
            state = states[step]
            action = actions[step]
            p_action = p_actions[step]
            next_state = next_states[step]
            print(
                f"Step {step:3d}:",
                f"s = [{', '.join(f'{i:2d}' for i in state)}],",
                f"a = {action:2d},",
                f"p_a = {p_action:4.2f},",
                f"{reward = :+3d},",
                f"sp = [{', '.join(f'{i:2d}' for i in next_state)}],",
            )

    def generate_episode(
        self, policy, summary=False, render=False, title=None, file_name=None
    ):
        """
        Generate an episode of interactions with the environment following a
        given policy.
        -----------
        Args:
            policy (callable):
                A policy function that selects actions based on the current
                state.
            summary (bool, optional, default=False):
                Flag to print an episode summary.
            render (bool, optional, default=False):
                Flag to render the episode.
            title (str, optional, default=None):
                Title for the rendered episode.
            file_name (str, optional, default=None):
                File name for saving the rendered episode.
        -----------
        Returns:
            tuple:
                A tuple containing lists of states, actions, action
                probabilities, rewards, and next states in the episode.
        """
        #
        self._fig_ax = None
        states = []
        actions = []
        p_actions = []
        rewards = []
        next_states = []
        #
        while True:
            state = self.state
            action, p_action = policy(state)
            states.append(state)
            actions.append(action)
            p_actions.append(p_action)
            #
            state, reward, is_state_terminal = self.step(action)
            #
            rewards.append(reward)
            next_states.append(state)
            if render:
                fig, ax = self.render()
            if is_state_terminal:
                break
        # Print summary of episode.
        if summary:
            self._print_episode_summary(
                states, actions, p_actions, rewards, next_states
            )
        # Show and save if requested.
        if render:
            fig.suptitle(title)
            if file_name is not None:
                fig.savefig(
                    file_name + ".pdf", bbox_inches="tight", pad_inches=0.1
                )
            #
        return states, actions, p_actions, rewards, next_states

    @property
    def map(self):
        """Returns the map of race track in 2D numpy.ndarray."""
        return self._map

    @property
    def n_action(self):
        """Returns number of actions."""
        return len(self._actions)

    @property
    def n_state(self):
        """
        Returns 1D array that gives dimension of each element of states.
        """
        return self._n_state

    @property
    def state(self):
        """Returns current state in a list."""
        return self._state.tolist()

    @classmethod
    def track_a(cls):
        """
        Create a Track instance with the track A map
        -----------
        Returns:
            Track:
                An instance of the Track class with the Track A map.
        """
        cell_types = cls.CELL_TYPES
        # Track A according to exercise 5.12.
        track_map = np.ones((32, 17), dtype=int) * cell_types["track"]
        track_map[:4, 0] = cell_types["gravel"]
        track_map[13:, 0] = cell_types["gravel"]
        track_map[:3, 1] = cell_types["gravel"]
        track_map[21:, 1] = cell_types["gravel"]
        track_map[0, 2] = cell_types["gravel"]
        track_map[28:, 2] = cell_types["gravel"]
        track_map[6, 10:] = cell_types["gravel"]
        track_map[7:, 9:] = cell_types["gravel"]
        track_map[-1, 3:9] = cell_types["start"]
        track_map[:6, -1] = cell_types["finish"]
        track = cls(track_map)
        return track

    @classmethod
    def track_b(cls):
        """
        Create a Track instance with the track B map
        -----------
        Returns:
            Track:
                An instance of the Track class with the Track B map.
        """
        cell_types = cls.CELL_TYPES
        # Track A according to exercise 5.12.
        track_map = np.ones((30, 32), dtype=int) * cell_types["track"]
        track_map[:-3, 0] = cell_types["gravel"]
        track_map[:-4, 1] = cell_types["gravel"]
        track_map[:-5, 2] = cell_types["gravel"]
        track_map[:-6, 3] = cell_types["gravel"]
        track_map[:-7, 4] = cell_types["gravel"]
        track_map[:-8, 5] = cell_types["gravel"]
        track_map[:-9, 6] = cell_types["gravel"]
        track_map[:-10, 7] = cell_types["gravel"]
        track_map[:-11, 8] = cell_types["gravel"]
        track_map[:-12, 9] = cell_types["gravel"]
        track_map[:-13, 10] = cell_types["gravel"]
        track_map[:4, 11] = cell_types["gravel"]
        track_map[7:-14, 11] = cell_types["gravel"]
        track_map[:3, 12] = cell_types["gravel"]
        track_map[8:-15, 12] = cell_types["gravel"]
        track_map[:1, 13] = cell_types["gravel"]
        track_map[9:-16, 13] = cell_types["gravel"]
        track_map[0, 14:16] = cell_types["gravel"]
        track_map[9, -2:] = cell_types["gravel"]
        track_map[10, -5:] = cell_types["gravel"]
        track_map[11, -6:] = cell_types["gravel"]
        track_map[12, -8:] = cell_types["gravel"]
        track_map[13:, -9:] = cell_types["gravel"]
        track_map[-1, :23] = cell_types["start"]
        track_map[:9, -1] = cell_types["finish"]
        track = cls(track_map)
        return track


class MonteCarloRL:
    """
    Class for Monte Carlo on and off policy Reinforcement Learning.
    -----------
    Attributes:
        env (Track):
            The environment in which RL takes place.
        epsilon (float):
            The exploration factor for epsilon-greedy policy.
        gamma (float):
            The discount factor for future rewards.
        _Q (numpy.ndarray):
            The action-value function, representing state-action values.
        _pi (numpy.ndarray):
            The target policy, mapping states to actions.
    -----------
    Methods:
        target_policy(state):
            Returns the action selected by the target policy for a given state
            and its probability (which is always 1.0).
        behavior_policy(state):
            Returns the action selected by the behavior policy for a given
            state and its probability.
        off_policy_learn(num_episodes):
            Performs off-policy learning and returns episode rewards.
        on_policy_learn(num_episodes):
            Performs on-policy learning and returns episode rewards.
    """

    def __init__(self, env):
        """
        Class constructor.
        ----------
        Args:
            env (Track):
                The environment in which RL takes place.
        """
        self.env = env
        self.epsilon = 0.1
        self.gamma = 1.0
        n_state = env.n_state
        n_action = env.n_action
        self._Q = np.random.normal(size=np.hstack((n_state, n_action))) - 1000
        self._pi = np.argmax(self._Q, axis=-1)

    def target_policy(self, state):
        """
        Returns the action selected by the target policy for a given state and
        its probability (which is always 1.0).
        ----------
        Args:
            state (list):
                The state.
        ----------
        Returns:
            tuple:
                A tuple containing the action selected by the target policy and
                its probability (that is always 1.0).
        """
        return self._pi[tuple(state)], 1.0

    def behavior_policy(self, state):
        """
        Returns the action selected by the epsilon-soft behavior policy for a
        given state and its probability.
        ----------
        Args:
            state (list):
                The state.
        ----------
        Returns:
            tuple:
                A tuple containing the action selected by the behavior policy
                and its probability.
        """
        greedy_action, _ = self.target_policy(state)
        p_greedy_action = 1 - self.epsilon + self.epsilon / self.env.n_action
        #
        if np.random.rand() < self.epsilon:
            # Non-greedy action, exploration.
            action = np.random.randint(self.env.n_action)
            if action == greedy_action:
                p_action = p_greedy_action
            else:
                p_action = self.epsilon / self.env.n_action
        else:
            # Greedy action, exploitation.
            action = greedy_action
            p_action = p_greedy_action
        return action, p_action

    def off_policy_learn(self, num_episodes):
        """
        Performs off-policy Monte Carlo learning, learns optimal policy and
        returns episode rewards.
        ----------
        Args:
            num_episodes (int):
                The number of episodes to run.
        ----------
        Returns:
            numpy.ndarray:
                An array containing the episode rewards.
        """
        env = self.env
        episode_rewards = np.zeros(num_episodes, dtype=int)
        Q = self._Q
        C = np.zeros_like(Q)
        for episode in tqdm(
            range(num_episodes), desc="Episodes", mininterval=0.5
        ):
            # Reset environment.
            env.reset()
            # Generate an episode.
            (
                states,
                actions,
                p_actions,
                rewards,
                _,
            ) = env.generate_episode(self.behavior_policy)
            # Iterate over episode backward and perform RL update.
            G = 0
            W = 1
            for state, action, p_action, reward in zip(
                reversed(states),
                reversed(actions),
                reversed(p_actions),
                reversed(rewards),
            ):
                state_action = tuple(state + [action])
                state = tuple(state)
                # Update action value.
                G = self.gamma * G + reward
                C[state_action] = C[state_action] + W
                Q[state_action] = (
                    Q[state_action]
                    + (G - Q[state_action]) * W / C[state_action]
                )
                # Update target policy.
                self._pi[state] = Q[state].argmax()
                if self._pi[state] != action:
                    break
                W = W / p_action
            #
            episode_rewards[episode] = sum(rewards)
        #
        self._Q = Q
        return episode_rewards

    def on_policy_learn(self, num_episodes):
        """
        Performs on-policy Monte Carlo learning, learns optimal policy and
        returns episode rewards.
        ----------
        Args:
            num_episodes (int):
                The number of episodes to run.
        ----------
        Returns:
            numpy.ndarray:
                An array containing the episode rewards.
        """
        env = self.env
        episode_rewards = np.zeros(num_episodes, dtype=int)
        Q = self._Q
        N = np.zeros_like(Q)
        for episode in tqdm(
            range(num_episodes), desc="Episodes", mininterval=0.5
        ):
            # Reset environment.
            env.reset()
            # Generate an episode.
            (
                states,
                actions,
                _,
                rewards,
                _,
            ) = env.generate_episode(self.behavior_policy)
            # Find first-visit indexes.
            indexes = np.unique(
                np.hstack((np.array(states), np.array(actions, ndmin=2).T)),
                return_index=True,
                axis=0,
            )[1]
            # Calculate returns.
            returns = np.cumsum(rewards[::-1])[::-1]
            # Iterate over episode and update actions and action values.
            for index in indexes:
                state = states[index]
                action = actions[index]
                G = returns[index]
                state_action = tuple(state + [action])
                state = tuple(state)
                # Update action value.
                N[state_action] += 1
                Q[state_action] = (
                    Q[state_action] + (G - Q[state_action]) / N[state_action]
                )
                # Update target policy.
                self._pi[state] = Q[state].argmax()
            #
            episode_rewards[episode] = sum(rewards)
        #
        self._Q = Q
        return episode_rewards


def test_env():
    """
    Run a test episode in the environment using random policy and displays
    episode summary.
    """
    track = Track.track_a()
    policy = MonteCarloRL(track).behavior_policy
    track.generate_episode(policy, summary=True)


def plot_episode_rewards(episode_rewards, title="", file_name=None):
    """
    Plot episode rewards history.
    ----------
    Args:
        episode_rewards (numpy.ndarray):
            An array of episode rewards.
        title (str, optional, default=""):
            Title for the plot.
        file_name (str, optional, default=None):
            File name for saving the plot, if None figure is not saved.
    """
    fig, ax = plt.subplots()
    ax.plot(episode_rewards)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_ylim([-500, 0])
    ax.set_xscale("log")
    fig.suptitle(title + " reward history")
    if file_name is not None:
        fig.savefig(file_name + ".pdf", bbox_inches="tight", pad_inches=0.1)


def train_and_simulate(
    num_episodes=200000, track_a=True, off_policy=True, save_figure=False
):
    """
    Train an RL agent and simulate episodes on a racetrack environment.
    ----------
    Args:
        num_episodes (int, optional, default=200000):
            The number of episodes for training.
        track_a (bool, optional, default=True):
            If True, Track (a) is used, else Track (b) is used.
        off_policy (bool, optional, default=True):
            If True, off-policy Monte Carlo RL is used, else on-policy variant
            is used.
        save_figure (bool, optional, default=False):
            Flag to save figures.
    """
    # Define the track and policy based on input arguments.
    if track_a:
        track = Track.track_a()
        track_name = "a"
    else:
        track = Track.track_b()
        track_name = "b"
    #
    MC = MonteCarloRL(track)
    if off_policy:
        episode_rewards = MC.off_policy_learn(num_episodes)
        policy_type = "off_policy"
    else:
        episode_rewards = MC.on_policy_learn(num_episodes)
        policy_type = "on_policy"

    # Plot the episode rewards.
    title = f"Track ({track_name}) {policy_type.replace('_', ' ')}"
    file_name = None
    if save_figure:
        file_name = f"exercise_5_11_track_{track_name}_{policy_type}_rewards"
    plot_episode_rewards(episode_rewards, title, file_name)
    # Perform some simulations.
    track.noise = False
    track.reset()
    if save_figure:
        file_name = f"exercise_5_11_track_{track_name}_{policy_type}_1"
    _ = track.generate_episode(
        MC.target_policy,
        summary=True,
        render=True,
        title=f"{title}, 1",
        file_name=file_name,
    )
    #
    track.reset()
    if save_figure:
        file_name = f"exercise_5_11_track_{track_name}_{policy_type}_2"
    _ = track.generate_episode(
        MC.target_policy,
        summary=True,
        render=True,
        title=f"{title}, 2",
        file_name=file_name,
    )


########## test section ################################################
if __name__ == "__main__":
    pass
    # Off policy track a.
    train_and_simulate(
        num_episodes=500000, track_a=True, off_policy=True, save_figure=True
    )
    # Off policy track b.
    train_and_simulate(
        num_episodes=500000, track_a=False, off_policy=True, save_figure=True
    )
    # On policy track a.
    train_and_simulate(
        num_episodes=500000, track_a=True, off_policy=False, save_figure=True
    )
    # On policy track b.
    train_and_simulate(
        num_episodes=500000, track_a=False, off_policy=False, save_figure=True
    )
    plt.show()
