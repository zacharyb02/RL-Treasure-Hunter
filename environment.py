import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt

class GridWorldEnv(gym.Env):
    def __init__(self, grid_size=10, num_monsters=3):
        super(GridWorldEnv, self).__init__()
        self.grid_size = grid_size
        self.num_monsters = num_monsters
        self.observation_space = spaces.Box(low=-grid_size, high=grid_size, shape=(4,), dtype=np.int32)
        self.action_space = spaces.Discrete(4)  # 4 actions: up, down, left, right
        self.reset()

    def reset(self):
        self.agent_pos = np.array([
            random.randint(0, self.grid_size - 1),
            random.randint(0, self.grid_size - 1)
        ])  
        self.goal_pos = np.array([self.grid_size - 1, self.grid_size - 1])  # Bottom-right
        self.monsters = []
        while len(self.monsters) < self.num_monsters:
            monster = np.array([random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)])
            if not any(np.array_equal(monster, m) for m in self.monsters) and not np.array_equal(monster, self.agent_pos) and not np.array_equal(monster, self.goal_pos):
                self.monsters.append(monster)
        return self.get_state()

    def get_state(self):
        # Return relative positions of the goal and closest monster
        closest_monster = min(self.monsters, key=lambda m: np.linalg.norm(self.agent_pos - m))
        return np.concatenate([
            self.agent_pos - self.goal_pos,  # Relative position to goal
            self.agent_pos - closest_monster  # Relative position to closest monster
        ])

    def step(self, action):
        # Store old position for distance calculation
        old_distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)

        # Move agent
        if action == 0:  # Up
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 1:  # Down
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)
        elif action == 2:  # Left
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 3:  # Right
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)

        # Check if reached the goal
        if np.array_equal(self.agent_pos, self.goal_pos):
            return self.get_state(), 100, True, {}

        # Check for proximity to any monster
        if self.is_adjacent_to_monster():
            return self.get_state(), -50, True, {}  # Large penalty for losing

        # Calculate new distance to the goal
        new_distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)

        # Reward for getting closer to the goal
        distance_reward = (old_distance_to_goal - new_distance_to_goal) * 10  # Amplify distance reward
        reward = -1 + distance_reward  # Base step penalty plus distance reward

        # Move monsters dynamically
        self.move_monsters()

        return self.get_state(), reward, False, {}

    def move_monsters(self):
        for monster in self.monsters:
            new_position = monster.copy()
            direction = random.choice([0, 1, 2, 3])  # Randomly choose a direction: up, down, left, right
            if direction == 0:  # Up
                new_position[0] = max(monster[0] - 1, 0)
            elif direction == 1:  # Down
                new_position[0] = min(monster[0] + 1, self.grid_size - 1)
            elif direction == 2:  # Left
                new_position[1] = max(monster[1] - 1, 0)
            elif direction == 3:  # Right
                new_position[1] = min(monster[1] + 1, self.grid_size - 1)

            # Ensure monsters do not overlap
            if not any(np.array_equal(new_position, m) for m in self.monsters) and not np.array_equal(new_position, self.agent_pos):
                monster[:] = new_position

    def is_adjacent_to_monster(self):
        # Check all adjacent squares
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:  # Ignore the agent's current position
                    check_pos = self.agent_pos + np.array([dx, dy])
                    if any(np.array_equal(check_pos, monster) for monster in self.monsters):
                        return True
        return False

    def render(self, mode='human'):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = '.'
        grid[self.agent_pos[0], self.agent_pos[1]] = 'A'  # Agent
        grid[self.goal_pos[0], self.goal_pos[1]] = 'G'  # Goal
        for monster in self.monsters:
            grid[monster[0], monster[1]] = 'M'  # Monsters
        print("\n".join([" ".join(row) for row in grid]))
        print()

    def render_image(self):
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        grid[self.agent_pos[0], self.agent_pos[1]] = 'A'  # Agent
        grid[self.goal_pos[0], self.goal_pos[1]] = 'G'  # Goal
        for monster in self.monsters:
            grid[monster[0], monster[1]] = 'M'  # Monsters

        # Plot the grid as an image
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xticks(range(self.grid_size + 1))
        ax.set_yticks(range(self.grid_size + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color='black', linestyle='-', linewidth=1)
        ax.imshow(np.zeros((self.grid_size, self.grid_size)), cmap='Greys', extent=(0, self.grid_size, 0, self.grid_size))

        # Annotate the cells
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                ax.text(y + 0.5, self.grid_size - x - 0.5, grid[x, y], ha='center', va='center', fontsize=12, color='red' if grid[x, y] == 'A' else 'black')

        plt.close(fig)
        return fig
