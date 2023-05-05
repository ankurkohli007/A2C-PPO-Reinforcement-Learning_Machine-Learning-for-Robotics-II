[Machine Learning for Robotics II](https://corsi.unige.it/en/off.f/2022/ins/60241)<br>
**Programmer:** [Ankur Kohli](https://github.com/ankurkohli007)<br>
[M.Sc Robotics Engineering](https://corsi.unige.it/corsi/10635)<br>
[University of Genoa (UniGe), Italy](https://unige.it/en)<br>

# Reinforcement Learning using A2C Algorithm & PPO Algorithm for Lunar Lander 

## Abstract 

This task is about Reinforcement Learning in which the task is to train a robot to land on the moon using reinforcement learning algorithms. The goal is to develop a robot that can autonomously navigate the lunar surface and safely land on a designated landing spot. For this task, **A2C** and **PPO** algorithms are used.

## Problem Statement 

The problem statement for **Lunar Lander** using RL (Reinforcement Learning) is to develop an AI agent that can successfully land a spacecraft on the moon's surface with minimal fuel consumption and without crashing. The agent needs to learn how to control the spacecraft's thrusters to navigate through the moon's gravitational field and adjust its velocity and orientation to achieve a safe landing. The task requires the agent to balance competing goals, such as reducing the spacecraft's speed and aligning its orientation with the landing site, while conserving fuel to ensure a successful landing. The agent's performance is measured by a reward function that provides feedback on its actions based on how close it is to the target landing site and how much fuel it has consumed. The goal of the RL agent is to learn an optimal policy that maximizes the cumulative reward over a given number of episodes, through trial-and-error interactions with the environment. The problem is challenging due to the high-dimensional state and action spaces, and the complex dynamics of the spacecraft and moon's environment.

## Problem Solution

The solution for the **Lunar Lander** problem using RL involves training an AI agent to learn how to control the spacecraft's thrusters to achieve a safe landing on the moon's surface. Here are the steps involved in developing an RL solution for this problem:

* **Define the state space:** The state space consists of all relevant information that the agent needs to make decisions. In the case of Lunar Lander, the state space may include the spacecraft's position, velocity, orientation, angular velocity, and fuel remaining.

* **Define the action space:** The action space consists of all possible actions that the agent can take. In the case of Lunar Lander, the action space may include firing the thrusters in different directions and adjusting the spacecraft's orientation.

* **Define the reward function:** The reward function provides feedback to the agent on its actions. In the case of Lunar Lander, the reward function may provide positive rewards for moving closer to the target landing site and negative rewards for crashing or using too much fuel.

* **Choose an RL algorithm:** There are various RL algorithms available, such as Q-learning, SARSA, A2C, PPO and policy gradients. Each algorithm has its strengths and weaknesses, and the choice depends on the specific problem and requirements.

* **Train the RL agent:** The RL agent interacts with the environment, observes its state, takes actions, and receives rewards. Over time, the agent learns an optimal policy that maximizes the cumulative reward.

* **Evaluate the RL agent:** After training, the RL agent is evaluated on a separate set of test episodes to assess its performance on unseen data.

* **Iterate and improve:** RL is an iterative process, and the RL agent can be further improved by adjusting the hyperparameters, modifying the reward function, or changing the algorithm.

In summary, the solution for the Lunar Lander problem using RL involves defining the state space, action space, and reward function, choosing an RL algorithm, training the RL agent, evaluating its performance, and iteratively improving it. Through this process, the RL agent can learn to control the spacecraft's thrusters to achieve a safe landing on the moon's surface with minimal fuel consumption and without crashing.

## Stable Baseline3 (SB3)

For this task Stable Baseline3 (SB3) library is used to accomplish the goal.

[Stable Baselines 3 (SB3)](https://stable-baselines3.readthedocs.io/en/master/) is a popular library for training and evaluating reinforcement learning models. SB3 also provides several utility functions for preprocessing and visualizing RL data, as well as a unified interface for defining and training RL models. Additionally, it supports parallel training on multiple CPU cores and can be easily integrated with popular RL benchmarks and environments, such as [OpenAI Gym](https://www.gymlibrary.dev/).

Overall, SB3 is a powerful and flexible library for training and evaluating RL models that has been widely adopted by the research community and industry practitioners alike.

## Parameter & Environment Information

This environment is part of the [Box2D environments](https://www.gymlibrary.dev/environments/box2d/).

* Action Space Discrete(4)
* Observation Shape (8,)
* Observation High [1.5 1.5 5. 5. 3.14 5. 1. 1. ]
* Observation Low [-1.5 -1.5 -5. -5. -3.14 -5. -0. -0. ]
* Import gymnasium.make("LunarLander-v2")

## Description

This environment is a classic rocket trajectory optimization problem. According to Pontryagin’s maximum principle, it is optimal to fire the engine at full throttle or turn it off. This is the reason why this environment has discrete actions: engine on or off.

There are two environment versions: discrete or continuous. The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector. Landing outside of the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt.

### Action Space

There are four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.

### Observation Space

The state is an 8-dimensional vector: the coordinates of the lander in ```x``` & ```y```, its linear velocities in ```x``` & ```y```, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.

### Rewards

Reward for moving from the top of the screen to the landing pad and coming to rest is about 100-140 points. If the lander moves away from the landing pad, it loses reward. If the lander crashes, it receives an additional -100 points. If it comes to rest, it receives an additional +100 points. Each leg with ground contact is +10 points. Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame. Solved is 200 points.

### Starting State

The lander starts at the top center of the viewport with a random initial force applied to its center of mass.

### Episode Termination

The episode finishes if:

* the lander crashes (the lander body gets in contact with the moon);
* the lander gets outside of the viewport (x coordinate is greater than 1);
* the lander is not awake. From the [Box2D docs](https://box2d.org/documentation/md__d_1__git_hub_box2d_docs_dynamics.html#autotoc_md61), a body which is not awake is a body which doesn’t move and doesn’t collide with any other body:

*When Box2D determines that a body (or group of bodies) has come to rest, the body enters a sleep state which has very little CPU overhead. If a body is awake and collides with a sleeping body, then the sleeping body wakes up. Bodies will also wake up if a joint or contact attached to them is destroyed.*

### Version History

* **v2:** Count energy spent and in v0.24, added turbulance with wind power and turbulence_power parameters

* **v1:** Legs contact with ground added in state vector; contact with ground give +10 reward points, and -10 if then lose contact; reward renormalized to 200; harder initial random push.

* **v0:** Initial version

In this taks version **v2** is used. Below is the code shows the version history of the LunarLander is **v2**. Also, **action** and **shape of observation**. 

```python
env = gym.make("LunarLander-v2", render_mode="human")

print("The Action inter is descrete {}".format(env.action_space.n))
print("Shape of Observation is {}".format(env.observation_space.sample().shape))
```
## Important Libraries To Install

```
pip install gym
```

```
pip install pyglet==1.5.27
```

```
pip install stable-baseline3
```

```
pip install "gymnasium[all]"
```
* **Note:** This task is performed on ***JupyterNotebook***. For the installation of ***JupyterNotebook*** [click here](https://github.com/ankurkohli007/Research_Track_II_Assignment_1_JupyterNotebook).  

## Strategy

#### Important Libraries

Firstly, import the required libraries as given below:

```python
import numpy as np
import imageio
import os
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
import gymnasium  as gym
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
import scipy.stats as stats
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv 
```

#### Baseline Model

Secondly, the **BaseLine Model** which refers to a simple model that serves as a reference point for comparing the performance of more complex models. It is typically the most basic model that can solve the problem at hand, and its performance provides a benchmark for evaluating the effectiveness of more sophisticated models.

In the case of the Lunar Lander problem, a baseline model may involve a simple algorithm with a basic feature representation of the state space, such as the spacecraft's position and velocity. The baseline model may use a straightforward reward function that provides a positive reward for landing on the target site and a negative reward for crashing or using too much fuel. Below is the code of baseline model of our task:

```python
rewards = []
obs = env.reset()
done = False
MAX_RUN = 10

for i in range(MAX_RUN):
    while not done:
        env.render()
        action_sample = env.action_space.sample()
        # let's take a step in the environment 
        obs, rwd, done, info ,_  = env.step(action_sample)
        rewards.append(rwd)
env.close()
print("Mean Reward after {} max run is {}".format(MAX_RUN, np.mean(np.array(rewards))))
```

The performance of the baseline model can be evaluated by measuring its success rate in landing the spacecraft on the target site and its fuel consumption. Once the baseline model's performance is established, more sophisticated models can be developed and compared to determine if they can improve upon the baseline performance.

Overall, a baseline model in Machine Learning Reinforcement Learning is an essential starting point for evaluating the effectiveness of more complex models and determining if the additional complexity is justified. It provides a reference point for comparing different algorithms and feature representations and can help researchers identify areas for further improvement.

#### Reinforcement Learning (RL) For Training The Model

Furthermore, RL can be used for training a model by treating the model as the agent and the data as the environment. The model can make predictions or take actions based on the data, and the rewards or penalties can be based on the accuracy of the predictions or the success of the actions. Below is the code of RL for training the model of our task:

```python
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True
```

#### Model Training for A2C & PPO Algorithms

Moreover, two RL algorithms A2C & PPO were used to accomplish the goal the task. Below is the code which shows the implementation of A2C algorithm: 

```python
env = make_vec_env("LunarLander-v2", n_envs=1,monitor_dir="log_dir_A2C_8/")

policy = "MlpPolicy"
n_steps = 100
learning_rate = 0.0003
batch_size = 256
n_epochs = 10

callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir="log_dir_A2C_8/")
# instantiate the agent 
model = A2C(policy, env, learning_rate = learning_rate, n_steps = n_steps,tensorboard_log="./TensorBoardLog/", verbose=1)
# train the agent
model.learn(total_timesteps=8000000, callback=callback)
```

Also, for PPO algorithm implementation the code remain the same only change the log directory such as ```env = make_vec_env("LunarLander-v2", n_envs=1,monitor_dir="log_dir_PPO_8/")``` and change the model where instantiate the agent such as ```model = PPO(policy, env, learning_rate = learning_rate, n_steps = n_steps,tensorboard_log="./TensorBoardLog/", verbose=1)```. Below code shows the full implementation of the PPO algorithm:   

```python
env = make_vec_env("LunarLander-v2", n_envs=1,monitor_dir="log_dir_PPO_8/")

policy = "MlpPolicy"
n_steps = 100
learning_rate = 0.0003
batch_size = 256
n_epochs = 10

callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir="log_dir_A2C_8/")
# instantiate the agent 
model = PPO(policy, env, learning_rate = learning_rate, n_steps = n_steps,tensorboard_log="./TensorBoardLog/", verbose=1)
# train the agent
model.learn(total_timesteps=8000000, callback=callback)
```
* **Note:** Hyperparameters such as *learning_rate*, *batch_size*, *n_steps*, *n_epochs*, and so on remains the same in both the cases. After model training a comparison between A2C & PPO were concluded according to the observation. Also, for both A2C & PPO algorithms model is trained for **50,00,000** **timesteps** & **80,00,000** **timesteps**  

#### Plotting 

In this part, trained model for 50,00,000 & 80,00,000 timesteps is ploted for both the proposed aforementioned algorithms. Below is the code defined for plotting of data:

```python
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """

    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=100)
    # Truncate x
    x = x[len(x) - len(y):]
    fig = plt.figure(title, figsize=(12,5))
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed A2C after 80,00,000 Timesteps")
    plt.grid()
    plt.show()

plot_results("log_dir_A2C_8")
```
* **Note:** For PPO algorithm, change the *log_directory* under *plot_results* command line such as ```plot_results("log_dir_PPO_8")```. There are different names for different timesteps such as for 50,00,000 ```log_dir_A2C``` & ```log_dire_PPO``` and for 80,00,000 ```log_dir_A2C_8``` & ```log_dir_PPO_8```.  

#### Evaluation 

Here evaluated the performance of a trained model in a simulated environment through a set of metrics that reflect the agent's ability to learn and make good decisions.

```python
env = make_vec_env("LunarLander-v2", n_envs=1,monitor_dir="log_dir_A2C_8")
model = A2C.load(path="log_dir_A2C_8/best_model.zip")
```

* **Note:** For PPO algorithm, change the *log_directory* under *env* command line such as ```log_dir_A2C_8```. Also, for model change ```A2C.load``` & in path add ```log_dir_A2C_8/best_model.zip```. Below is the code for PPO algorthm:

```python
env = make_vec_env("LunarLander-v2", n_envs=1,monitor_dir="log_dir_PPO_8")
model = PPO.load(path="log_dir_PPO_8/best_model.zip")
```
#### Stable Baseline 3 Evaluation Function

Stable Baselines 3 (SB3) is a popular library for training and evaluating reinforcement learning models. SB3 provides several evaluation functions that can be used to assess the performance of a trained RL model. For this task, **evaluate_policy** function is used to accomplish the desired goal.

**evaluate_policy**: This function evaluates a given policy (i.e., the learned behavior of the RL agent) in a given environment over a specified number of episodes. It returns the mean and standard deviation of the total rewards obtained by the policy, as well as the mean and standard deviation of the episode lengths. The code below is shows how the aforementioned function is used for the evaluation of the trained model.

```python
mean_reward, std_reward = evaluate_policy(model, env,n_eval_episodes=10, render=True, deterministic=True)
print("Mean & Std Reward after {} max run is {} & {}".format(10,mean_reward, std_reward)) 
```

The above code provides the uses of the *evaluate_policy* function from the Stable Baselines 3 library to evaluate a trained RL model (model) in a given environment (env). Below is the brief description of the code:

* ```evaluate_policy(model, env, n_eval_episodes=10, render=True, deterministic=True)```; this line of code evaluates the given policy (model) in the given environment (env) over a specified number of episodes ```(n_eval_episodes=10)```.

* ```mean_reward, std_reward = evaluate_policy(model, env,n_eval_episodes=10, render=True, deterministic=True)```; this line of code stores the mean and standard deviation of the total rewards obtained by the policy in the variables ```mean_reward``` and ```std_reward```, respectively. These values are calculated by the ```evaluate_policy``` function.

* ```print("Mean & Std Reward after {} max run is {} & {}".format(10,mean_reward, std_reward))```; this line of code prints the mean and standard deviation of the total rewards obtained by the policy over the specified number of episodes (```n_eval_episodes=10```). The values are displayed in the message "Mean & Std Reward after 10 max run is <mean_reward> & <std_reward>".

## Comaprison between PPO & A2C Algorithms

From the aforementioned task implementation, general comparsion of the two stated algorithms is concluded below:

* PPO is a policy optimization method that aims to maximize the expected reward of a policy while ensuring that the update is not too far from the previous policy. Whereas, A2C is an on-policy method that learns both a policy and a value function. The advantage of A2C is that it can use the value function estimate to reduce the variance of the policy gradient.

* PPO is known for its stability, while A2C can learn faster and more efficiently.

* Both algorithms can perform well in the Lunar Lander environment, but their performance can depend on various factors such as hyperparameters used for training.

* A2C can suffer from high variance in the policy gradient estimate, which can lead to unstable learning and poor performance.

* Both algorithms can struggle to learn long-term dependencies in the Lunar Lander environment and may require additional techniques such as recurrent networks or curriculum learning to overcome this limitation.

## Limitations of PPO & A2C Algorithms 

From the aforementioned task implementation, general limitation of the two stated algorithms is concluded below:

* PPO and A2C can suffer from slow convergence and require careful tuning of hyperparameters to achieve good performance.

* Both algorithms can be sensitive to the choice of network architecture and the quality of the state representation.

* PPO can be prone to overfitting, especially if the reward signal is sparse or noisy.

* In general, PPO tends to be more stable than A2C, while A2C can achieve  higher scores in some situations.

* The choice between PPO and A2C will depend on the specific requirements of the task and the trade-offs between stability and speed of learning, as well as other factors such as network architecture and hyperparameter tuning.

## References 

* Mnih, V., Badia, P. A.,  Mirza, M., Graves, A., Harley, T., Lillicrap, P. T., Silver, D., & Kavukcuoglu, K. (2017). Proximal Policy Optimization Algorithms. In OpenAI.

* Schulman, J., Wolski, F.,  Dhariwal, P., Radford, A., & Klimov, O. (2016). Asynchronous Methods for Deep Reinforcement Learning. Proceedings of the 33rd International Conference on Machine Learning, New York, NY, USA, Vol. 40.

