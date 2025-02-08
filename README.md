# deep-reinforcement-learning-ppo-dpo

# Reinforcement Learning with Proximal Policy Optimization (PPO) and Direct Preference Optimization (DPO)

This project explores advanced reinforcement learning techniques for solving control tasks in the **Hopper** environment, using **Proximal Policy Optimization (PPO)** and **Direct Preference Optimization (DPO)**. These methods demonstrate how reinforcement learning models can be optimized based on both engineered reward functions and human preference data. This project is built on deep learning methods for policy learning and optimization.

---

## Project Overview

The main focus of this project is to:
1. Train a reinforcement learning agent to complete tasks in the Hopper environment using **Proximal Policy Optimization (PPO)**.
2. Apply **Direct Preference Optimization (DPO)** to optimize policies directly from human preference data without traditional reward functions.

---

## How to Use the Project

Everything you need is included in the `reinforcement_learning.zip` file:
- Fully functioning project
- Environment setup (`environment_CUDA.yml`)
- Scripts and Jupyter notebooks
- Datasets and outputs

Simply download and extract the zip file to get started!

---

## Algorithms Implemented

### **Proximal Policy Optimization (PPO)**
- **Objective**: Learn a policy by maximizing expected rewards while ensuring stability through clipped updates.
- **Implementation**: Uses a deep neural network to model the policy and value function. Trains on a predefined reward function in the Hopper environment to improve task completion performance.
  
### **Direct Preference Optimization (DPO)**
- **Objective**: Skip reward learning by optimizing a model directly on human preference data.
- **Environment**: Hopper, using human preference-labeled trajectories.
- **Methodology**: A deep learning-based policy network is fine-tuned to prioritize actions ranked higher by human raters.

---

## Environment Details

- **Hopper Environment** (from the MuJoCo simulator):  
  A reinforcement learning control task where an agent learns to balance and hop forward.
  
- **Preference Data**:  
  Contains human-labeled comparisons of different policy rollouts to guide training.

---

## Performance Results

#### **Proximal Policy Optimization (PPO)**  
- The **orange curve (no early termination)** shows better performance, achieving higher returns overall compared to the **blue curve (early termination)**.
- Early termination slows down learning and increases variability, suggesting that removing early termination allows for more stable long-term policy optimization.
- Both policies eventually flatten out at different levels, with the policy without early termination maintaining a higher average return and lower volatility compared to the policy with early termination.

#### **Direct Preference Optimization (DPO)**
- The DPO model starts with a sharp increase in performance but shows a decline in returns after iteration 20.
- This suggests that DPO may require further tuning (e.g., more preference data, better initialization) to sustain performance over time.
- Despite this, the model still achieves relatively high early returns, demonstrating its ability to optimize based solely on preferences without a reward function.

---

## Key Insights

- **Deep learning** models for policy learning in PPO and DPO can show robust performance but may require careful tuning to maintain long-term stability.
- **PPO**: Policies trained with PPO using engineered rewards showed stable learning but were sensitive to early termination conditions.
- **DPO**: Policies optimized from human preference data effectively bypassed the need for complex reward design, though they may require further tuning to maintain performance over time.

---

## Tools and Technologies

- **Python**: Core programming language.
- **PyTorch**: Deep learning library for neural network-based policy implementation.
- **Gymnasium (MuJoCo)**: Environment for reinforcement learning simulations.
- **Matplotlib**: Visualization of training results.
- **Deep Learning**: Used to implement and optimize policies through neural networks.

---

## About Me

I'm a data scientist and deep learning practitioner specializing in reinforcement learning and policy optimization. This project highlights my expertise in:
- Developing deep learning-based models for decision-making tasks.
- Applying reinforcement learning techniques such as PPO and DPO to real-world control problems.
- Using neural networks to optimize policies through both reward-based and preference-based approaches.
  
