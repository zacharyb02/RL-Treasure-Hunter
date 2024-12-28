 # RL-Treasure-Hunter

RL-Treasure-Hunter is a reinforcement learning project that implements an AI agent capable of navigating a 10x10 grid world to find a treasure while avoiding dynamic obstacles (monsters). The agent uses the Q-Learning algorithm, implemented via the Stable Baselines3 library. Thsi project is made for the AI Internship Technical test for Jumbo Mana

## Project Structure

```
RL-Treasure-Hunter/
├── frames/                 # Frames captured during testing
│   ├── frame_0000.png
│   ├── frame_0001.png
│   └── ...
├── model/                  # Saved models
│   └── dqn_agent.zip       # Trained model
├── src/                    # Source code
│   ├── environment.py      # Environment definition
│   ├── main.py             # Training script
│   └── test.py             # Testing script
├── video/                  # Videos generated during testing
│   └── agent_performance.mp4
├── README.md               # Project overview and instructions
├── requirements.txt        # Python dependencies
```

## Requirements

The project requires Python 3.8+ and the following libraries:

- Stable Baselines3
- MoviePy
- Matplotlib
- NumPy
- Gym

Install dependencies using:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Train the Agent

To train the agent, run the `main.py` script located in the `src` directory:

```bash
python src/main.py
```

This script initializes the GridWorld environment, trains the agent using DQN, and saves the trained model to the `model/` directory as `dqn_agent.zip`.

### 2. Test the Agent

To test the trained agent and visualize its performance, run the `test.py` script:

```bash
python src/test.py
```

This script loads the trained model, captures the agent's path as frames, and saves them in the `frames/` directory.

### 3. Generate a Video

After testing, the `test.py` script also generates a video showcasing the agent's performance, saved in the `video/` directory as `agent_performance.mp4`.

## Features

- **Dynamic Obstacles:** The environment includes dynamic monsters with configurable movement patterns.
- **Random Start Position:** The agent starts from a random position on the grid in every episode.
- **Training Metrics:** Logs rewards, losses, and other metrics for evaluation.
- **Visualization:** Generates videos to visualize the agent's behavior during testing.

## Code Structure

- `environment.py`: Defines the GridWorld environment, including the grid, monsters, agent, and reward structure.
- `main.py`: Handles training the agent using DQN and saving the trained model.
- `test.py`: Loads the trained model, tests the agent, captures frames, and generates a performance video.

## Results

- **Evaluation Reward:** Mean: 117.20 ± 84.38
![alt text](https://github.com/zacharyb02/RL-Treasure-Hunter/blob/main/episode_rewards.png?raw=true)


