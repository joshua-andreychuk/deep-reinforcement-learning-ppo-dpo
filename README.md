# deep-reinforcement-learning-ppo-dpo-rlhf  
**Reinforcement Learning with Proximal Policy Optimization (PPO), Direct Preference Optimization (DPO), and Reinforcement Learning from Human Feedback (RLHF)**  

This project explores advanced reinforcement learning techniques for solving control tasks in the Hopper environment, using engineered reward functions, human preference data, and learned reward models.  

## Project Overview  
The main focus of this project is to:  
- Train a reinforcement learning agent to complete tasks in the Hopper environment using **Proximal Policy Optimization (PPO)**.  
- Apply **Direct Preference Optimization (DPO)** to optimize policies directly from human preference data without traditional reward functions.  
- Implement **Reinforcement Learning from Human Feedback (RLHF)**: learn a reward model from human pairwise preferences, then fine-tune a PPO policy using that learned reward.  

## Algorithms Implemented  

### Proximal Policy Optimization (PPO)  
- **Objective**: Learn a policy by maximizing expected rewards while ensuring stability through clipped updates.  
- **Implementation**: Uses a deep neural network to model the policy and value function. Trains on a predefined reward function in the Hopper environment to improve task completion performance.  

### Direct Preference Optimization (DPO)  
- **Objective**: Skip reward learning by optimizing a model directly on human preference data.  
- **Environment**: Hopper, using human preference–labeled trajectories.  
- **Methodology**: A deep learning–based policy network is fine-tuned to prioritize actions ranked higher by human raters.  

### Reinforcement Learning from Human Feedback (RLHF)  
- **Objective**: First learn a differentiable reward model from human trajectory comparisons, then optimize a policy with PPO on that learned reward.  
- **Methodology**:  
  1. **Reward model**: a small PyTorch network (state+action → hidden layer → sigmoid) trained on Bradley–Terry cross-entropy over summed trajectory returns (100 k updates).  
  2. **Custom Gym wrapper**: replaces MuJoCo’s reward with model predictions while logging the ground-truth reward.  
  3. **PPO fine-tuning**: 1 M environment steps on the learned reward, evaluated every 1 k steps over 10 episodes.  

## Environment Details  
**Hopper Environment** (from the MuJoCo simulator):  
A reinforcement learning control task where an agent learns to balance and hop forward.  

**Preference Data**:  
Contains human-labeled comparisons of different policy rollouts to guide training for both DPO and reward-model learning.  

## Performance Results  

### Proximal Policy Optimization (PPO)  
(ppo_hopper.png)  
The orange curve (no early termination) shows better performance, achieving higher returns overall compared to the blue curve (early termination). Early termination slows down learning and increases variability, suggesting that removing early termination allows for more stable long-term policy optimization. Both policies eventually flatten out at different levels, with the policy without early termination maintaining a higher average return and lower volatility compared to the policy with early termination.  

### Direct Preference Optimization (DPO)  
(hopper_dpo.png)  
The DPO model starts with a sharp increase in performance but shows a decline in returns after iteration 20. This suggests that DPO may require further tuning (e.g., more preference data, better initialization) to sustain performance over time. Despite this, the model still achieves relatively high early returns, demonstrating its ability to optimize based solely on preferences without a reward function.  

### Reinforcement Learning from Human Feedback (RLHF)  
(hopper_rlhf.png)  
The blue curve (“RLHF (original)”) is PPO trained on the true MuJoCo reward, and the orange curve (“RLHF (learned)”) is PPO trained on my learned reward model. The learned-reward policy steadily improves to around 600 return, demonstrating that a preference-trained reward can drive continuous-control learning—albeit at a lower absolute scale than the engineered baseline.  
> **Note:** my reward model is trained with a Bradley–Terry (pairwise cross-entropy) objective to rank trajectories, not to regress true return magnitudes, so outputs naturally compress toward the middle of the scale. This compression explains why PPO on the learned reward tops out below the true-reward baseline.  

## Key Insights  
- **PPO**: Policies trained with PPO using engineered rewards showed stable learning but were sensitive to early termination conditions.  
- **DPO**: Policies optimized from human preference data effectively bypass the need for complex reward design, though they may require further tuning to maintain performance over time.  
- **RLHF**: Learning a reward model from preferences and then fine-tuning with PPO is a modular pipeline—once you have a good reward model, any off-the-shelf RL algorithm can optimize it. The compressed scale of learned rewards means the policy still improves reliably, but additional preference data or larger model capacity may be needed to close the performance gap with the true MuJoCo reward.  

## Tools and Technologies  
- **Python**: Core programming language.  
- **PyTorch**: Deep learning library for neural network–based policy and reward modeling.  
- **Gymnasium (MuJoCo)**: Environment for reinforcement learning simulations.  
- **Matplotlib**: Visualization of training results.  
- **Deep Learning**: Used throughout for both policy/value networks and reward modeling.  

## About Me  
I’m a data scientist and deep learning practitioner specializing in reinforcement learning and policy optimization. This project highlights my expertise in:  
- Developing deep learning–based models for decision-making tasks.  
- Applying reinforcement learning techniques such as PPO, DPO, and RLHF to real-world control problems.  
- Using neural networks to optimize policies through both reward-based and preference-based approaches.  
