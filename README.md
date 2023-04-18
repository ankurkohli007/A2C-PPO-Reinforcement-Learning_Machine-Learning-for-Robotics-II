[Machine Learning for Robotics II](https://corsi.unige.it/en/off.f/2022/ins/60241)<br>
**Programmer(s):** [Ankur Kohli](https://github.com/ankurkohli007)<br>
[M.Sc Robotics Engineering](https://corsi.unige.it/corsi/10635)<br>
[University of Genoa (UniGe)](https://unige.it/en)<br>



<br>
<br>
Reinforcement learning (RL) has shown great potential for robotic applications, particularly in tasks where the robot must learn from interaction with the environment and adapt to changing conditions. In RL, the robot learns a policy or set of behaviors that maximize a reward signal provided by the environment, rather than being explicitly programmed with a set of rules or actions.

One of the main advantages of RL in robotics is its ability to learn from experience, allowing the robot to adapt to different environments and tasks without the need for manual intervention or programming. RL algorithms can also handle continuous state and action spaces, making them well-suited to many robotic tasks that require precise and continuous control.

Some examples of RL in robotics include:

Robotic manipulation: RL can be used to teach robots to manipulate objects, such as picking and placing objects in a warehouse or assembling components in a manufacturing plant.

Autonomous navigation: RL can be used to teach robots to navigate through complex environments, such as autonomous vehicles that must navigate through traffic or drones that must navigate through cluttered spaces.

Human-robot interaction: RL can be used to teach robots to interact with humans in a natural and intuitive way, such as recognizing and responding to gestures or facial expressions.

Control and optimization: RL can be used to optimize the control of robotic systems, such as balancing a bipedal robot or controlling the motion of a robotic arm.

Lunar Lander RL (Reinforcement Learning) is a popular application of machine learning in robotics. It involves training a robot to land on the moon using reinforcement learning algorithms. The goal is to develop a robot that can autonomously navigate the lunar surface and safely land on a designated landing spot.<br>


In Lunar Lander RL, the robot learns through trial and error. The reinforcement learning algorithm provides the robot with feedback on its actions, rewarding it for good decisions and punishing it for bad ones. Over time, the robot learns to make better decisions based on the feedback it receives.<br>


The Lunar Lander RL problem is a challenging one, as it requires the robot to navigate a complex environment with limited resources. The robot must carefully manage its fuel, adjust its trajectory to avoid obstacles, and make split-second decisions to avoid crashing.<br>


Several techniques have been used to tackle the Lunar Lander RL problem, including deep reinforcement learning, policy gradient methods, and Q-learning. These techniques have been used to develop robots that can land on the moon with high accuracy and efficiency.<br>


Overall, Lunar Lander RL is an exciting and challenging problem in robotics that has the potential to advance our understanding of machine learning and autonomous navigation.


##### Comaprison between PPO & A2C

* PPO is a policy optimization method that aims to maximize the expected reward of a policy while ensuring that the update is not too far from the previous policy. Whereas, A2C is an on-policy method that learns both a policy and a value function. The advantage of A2C is that it can use the value function estimate to reduce the variance of the policy gradient.

* PPO is known for its stability, while A2C can learn faster and more efficiently.

* Both algorithms can perform well in the Lunar Lander environment, but their performance can depend on various factors such as hyperparameters used for training.

* A2C can suffer from high variance in the policy gradient estimate, which can lead to unstable learning and poor performance.

* Both algorithms can struggle to learn long-term dependencies in the Lunar Lander environment and may require additional techniques such as recurrent networks or curriculum learning to overcome this limitation.
