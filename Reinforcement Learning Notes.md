## Reinforcement Learning Notes

**[Resource1: Lectures, Books, Surveys and Thesis of Reinforcement Learning](https://github.com/aikorea/awesome-rl)**

**[Resource2: An Outsider’s Tour of Reinforcement Learning](http://www.argmin.net/2018/06/25/outsider-rl/)**



#### <font color='seagreen'> Basic Notations and Elements</font>

- **Policy**: defines behavior of the learning agent at a given time
- **Reward Function**: defines the goal, or reward mapped to each perceived state
- **Value Function**: define the total reward in long run, our final goal to optimize

**<font color = 'steelblue'> the most important component in almost all RL problems is estimating *values*</font>**

Some RF methods don’t necessarily require to search for value functions, for example, the ***evalutionary methods***, including genetic algorithms, genetic programming, simulated annealing and so on.

- **Model**: to generate environment, or mimic the behavior of environments. For example, given a state and action, the model might predict the resultant next state and next reward



#### <font color='seagreen'>Evaluative Feedback</font>

**<font color='olive'>N-armed Bandit Problem</font>**

- **Denotions**
  - *value*: expected or mean reward given a selected action
  - *greedy action*: maintain estimates of each action values for each step, and there is always a greates action win over other actions
  - *Exploiting*: select the greedy action
  - *Exploring*: select the nongreedy action

**<font color='steelblue'>Exploitation is the right thing to do to maximize the expected reward on one play, but exploration may produce greater total reward in the long run</font>**



**<font color='olive'>Action-Value Methods</font>**

- **Denotions**

  - *a*: an action
  - $Q^{*}(a)$: true (actual) value of action *a*
  - $Q_{t}(a)$: estimated value of action *a*
  - *t*: rount of play
  - $k_{a}$: the times of action a has been chosen
  - $r_{k_{a}}$: the reward of action a after it has been chosen k times

  Then a simple way to estimate $Q_{t}(a)$ is:

  $Q_{t}(a)$ = $\frac{r_{1}+r{2}+…+r_{k_{a}}}{k_{a}}$

  If $k_{a}=0$, then default value is $Q_{t}(a)=0$;

  if $k_{a} \rightarrow \infty$, by the large number law, $Q_{t}(a) \rightarrow Q^{*}(a)$

- **$\epsilon$-greedy vs. greedy**

  design an experiment to asses the effectiveness of these two methods

  `Settings`:

  - 2000 n-armed bandit games, 1000 plays for each game
  - n=10, for each game
  - For each action, reward is randomly generated from $N~(Q^{*}(a),1)$ for 1000 round of plays
  - $Q^{*}(a)$ is randomly generated from $N(0,1)$ 2000 times for 2000 different games
  - record the performance after each 1000 plays
  - set $\epsilon$=0, 0.01, and 0.1

The code has been recaptured by myself. Here are the analytical results of $\epsilon$-greedy mrthods.

![epsilon_method](Reinforcement Learning Notes.assets/epsilon_method-9526924.png)

the above image depicts the total reward accumulated under different $\epsilon$ with 50 games, and 4000 rounds in each game.

![optimal_action](Reinforcement Learning Notes.assets/optimal_action.png)

This image depicts the optimal action’s accumulated reward. Other parameters setting is the same.

This image depicts the optimal action’s accumulated reward. Other parameters setting is the same.

**<font color ='steelblue'>The advantage of this method depends on the task</font>**

- If the variance is >1, it will take more exploration to find out the optimal action

- If variance=0, then the greedy method without $\epsilon$ will accumulate the optimal value

- **<font color = 'cadetblue'>another complex situation is that the true value of each action changes overtime (nonstationary). And this is the commonly encountered situation in reinforcement learning.</font>**

  

**<font color='olive'>Softmax Action Selection</font>**

The probability of choosing each action now is given by a softmax method. The common method uses Gibbs or Boltzmann distribution.

It chooses action *a* on *t*th play with prob $\frac{e^{Q_{t}(a)/\tau}}{\sum_{b=1}^{n}e^{Q_{t}(b)/\tau}}$

Parameter $\tau$ is called **Tempreture**

- High temperature will cause actions to be all equiprobable. When $\tau \rightarrow \infty$, $prob \rightarrow \frac{1}{n}$. 
- Low temperature will cause actions to differ by their estimated values. When $\tau \rightarrow 0$, the method will become greedy.

Whether <font color='steelblue'> epsilon-greedy</font> or <font color='steelblue'>softmax</font> is unclear. It depends on the task.



**<font color='olive'>Evaluations Versus Instructions</font>**

- The reward received after each action gives some information about how good the action was, but it **says nothing at all about whether the action was correct or incorrect, that is, about whether or not it was best**
- RL sharply contrasts with supervised learning. **In supervised learning, there is no need to try various actions.** Feedback from the environment directly indicates what the action should have been. **Feedback is independently of actions taken.**

***<font color='skyblue'>Two examples to illustrat differnece in Evaluative and Instructive</font>***

- Suppose there are 100 different actions. If you select action 32, evaluation will tell you the score you get from action 32. And you have to try various actions to find out the optimal strategy. Instruction would say what other action, say action number 67, would actually have been correct.

- Evaluative training and instructive training use different optimization algorithms.

  Instructive training use algo like Gradient Descent, to tell the algorithm where to go to search the parameter space.

  Evaluative training use other algos to explore around space for optimization. Typical examples are Robbins–Monro and the Kiefer–Wolfowitz stochastic approximation algorithms.

***<font color='steelblue'>Binary Bandit Problem</font>***

…...



**<font color='olive'>Incremental Implementation</font>**

The action-value methods mentioned above use sample average to estimate action values:

$Q_{t}(a)=\frac{r_{1_{a}}+r_{2_{a}}+…+r_{k_{a}}}{k_{a}}$

A problem with this straightforward method to estimate action values is that it increases the memory usage without bound when time t increases.

**Incremental Implementation** can solve this problem. This method can estimate action values without t appearing in the estimation.

**Denotions**:

- $Q_{k}$: average of first k rewards for some action.
- $Q_{k_{a}}$: the reward of action a at **kth** play. <font color='lightskyblue'>**Don’t mess up this with the above one.**</font>

So the **incremental implementation** goes as follow:

$Q_{k+1} = \frac{1}{k+1}\sum_{i=1}^{k+1} r_{i}$

$= \frac{1}{k+1}(r_{k+1}+\sum_{i=1}^{k}r_{i})$

$=\frac{1}{k+1}(r_{k+1}+kQ_{k}+Q_{k}-Q_{k})$

$=\frac{1}{k+1}(r_{k+1}+(k+1)Q_{k}-Q_{k})$

$=Q_{k}+\frac{1}{k+1}(r_{k+1}-Q_{k})$

And the above method can be concluded as:

$NewEstimate \leftarrow OldEstimate+Stepsize(Target - OldEstimate)$

We denote the $Stepsize$ as $\alpha$.

When $\alpha=\frac{1}{k}$, then this is the sample average method.

**<font color='steelblue'>This is still the sample average method, but with a memory-saving method, which is incremental implementation</font>**



**<font color='olive'>Tracking a Nonstationary Problem</font>**

The above method is appropriate for stationary environment. For nonstationary environment, we can make $\alpha$ equal to a constant. **When $\alpha=constant$, the recent rewards will be given more weights than those past rewards.**

$Q_{k}=Q_{k-1}+\alpha(r_{k}-Q_{k-1})$

$Q_{k}=\alpha r_{k}+(1-\alpha)(Q_{k-1})$

$Q_{k}=\alpha r_{k}+(1-\alpha)(Q_{k-2}+\alpha(r_{k-1}-Q_{k-2}))$

$Q_{k}=\alpha r_{k}+\alpha(1-\alpha)r_{k-1}+(1-\alpha)^{2}Q_{k-2}$

…….

$Q_{k}=\alpha r_{k}+\alpha(1-\alpha)r_{k-1}+\alpha(1-\alpha)^{2}r_{k-2}+…+\alpha(1-\alpha)^{k-i}r_{i}+…+\alpha(1-\alpha)^{k-1}r_{1}+(1-\alpha)^{k}Q_{0}$

$Q_{k}=(1-\alpha)^{k}Q_{0}+\sum_{i=1}^{k}\alpha(1-\alpha)^{k-i}r_{i}$

We call this weighted average because $(1-\alpha)^{k}+\sum_{i=1}^{k}\alpha(1-\alpha)^{k-i}=1$

This is sometimes also called ***exponetial, recency-weighted average***

**<font color='steelblue'>Sometimes stepsize vary with timestep.  </font> **$\alpha_{k}(a)$ **<font color='steelblue'>denotes the step-size parameter used to process the reward received after the kth selection of action a. </font>**

However, when we choose various step-size, convergence is not guaranteed. Sample average ($\alpha=\frac{1}{k}$ gurantees convergence to true action value by LLN). 

**Two conditions required to guarantee convergence:**

- $\sum_{k=1}^{\infty}\alpha_{k}(a)=\infty$
- $\sum_{k=1}^{\infty}\alpha_{k}^{2}(a)<\infty$

Both conditions are met in sample average case ( $\alpha=\frac{1}{k}$ ), but the second condition is not met for constant step-size case ( $\alpha=const.$ ). And actually, this is desired in nonstationary case.

- The first condition is required to guarantee that the steps are large enough to eventually overcome any initial conditions or random fluctuations.
- The second condition guarantees that eventually the steps become small enough to assure convergence.



**<font color='olive'>2.6 Optimistic Initial Values</font>**

All above methods have ignored the influence of initial action-value estimates $Q_{t=1}(a)$. In statistics, these methods are biased by their initial values.

<font color='olivedrab'>***What are brought by this bias***</font>

- Bias is permanent when $\alpha$ is constant but decrease over time.
- Bias disappears when $\alpha=\frac{1}{k}$ (sample average) after all actions have been selected as least once.
- **Upside**: provide a prior knowledge about what level of rewards can be expected
- **Downside**: initial estimates become another set of parameters need to be estimated
- When initial values are set far away from true value, this will encourage exploration, we call this strategy **optimistic initial values**. But this only works for ***stationary*** problem. 



**<font color='olive'>2.7 Upper-Confidence-Bound Action Selection</font>**

When we do exploring, it would be better if we can consider actions’ uncertainties and potential for being optimal. One effective way of selecting action is by:

$A_{t}=argmax_{a}(Q_t(a)+c\sqrt{\frac{lnt}{N_{t}(a)}})$

- $A_t$ is the selection
- $lnt$ is the natural logarithm of time 
- $N_t(a)$ is the times of taking action a before time t, the same as $k_a$ we used before
- $c$ controls the degree of exploration

when $N_t(a)=0$, the corresponding action will be chosen, because an action never been explored has greatest potential.



**<font color='olive'>2.8 Gradient Bandits</font>**

In above methods, we assign probabilities to each action during each play. **Now we consider a prefenrence $H_t(a)$ to decide the probability $\pi_t(a)$ of selecting action a at time t.**

$Pr(A_t=a)=\pi_t(a) = \frac{e^{H_t(a)}}{\sum_{b}^{n}e^{H_t(b)}}$

To learn these preferences, we can update preferences after each step by:

- $H_{t+1}(a) = H_{t}(a)+\alpha(R_t-\bar{R_{t}})(1-\pi_{t}(A_t))$  for $a = A_{t}$  (2.10)
- $H_{t+1}(a) = H_{t}(a)-\alpha(R_t-\bar{R_{t}})\pi_{t}(A_t)$   $\forall a \ne A_{t}$    (2.10)

where $\alpha$ is the step-size (depends on stationary or nonstationary problem), $\bar{R}$ is the average of all rewards up to time t.

**<font color='deepskyblue'> Based on the preference-updating function, we can tell that, once the selected action increases current reward, then the preference of selecting this action will increase also. Vice versa.</font>**



***<font color='darkolivegreen'>One can gain a deeper insight into above algorithm by understanding stochastic approximation of gradient ascent. In the perspective of SGA, the algorithm goes like:</font>***

- $H_{t+1}(a) = H_{t}(a)+\alpha\frac{\partial\mathbb{E}[R_t]}{H_{t}(a)}$  (2.11)

- where   $\mathbb{E}[R_t]=\sum_{b}^{n}q(b)\pi(b)$

 $q(b)$ is the true reward get from action b. However, we don’t know $q(b)$. Fortunately, **the updates in algorithm 2.10 equals to algorithm 2.11 in expected values.**



***<font color='darkolivegreen'>Mathematical Proof</font>***

$H_{t+1}(a) = H_{t}(a)+\alpha\frac{\partial\mathbb{E}[R_t]}{\partial H_{t}(a)}$

Focus on $\frac{\partial\mathbb{E}[R_t]}{H_{t}(a)}$

$\frac{\partial\mathbb{E}[R_t]}{H_{t}(a)} = \frac{\partial\sum_{b}^{n}q(b)\pi(b)}{\partial H_{t}(a)}$ 

=$\sum_{b}^{n}q(b) \frac{\partial \pi(b)}{\partial H_t(a)}$     $q(b)$ can be extracted before because true reward is fixed. $\pi(b)$ is a function of $H_t(b)$

=$\sum_{b}^{n}(q(b)-X_t) \frac{\partial \pi(b)}{\partial H_t(a)}$    

…...



**<font color='olive'>2.9 Associative Search</font>**

…...







#### <font color='seagreen'>Chapter 3. Finite Markov Decision Process</font>

**<font color='olive'>3.1 The Agent-Environment Interface</font>**

- The learner and decision-maker is called *agent*
- The thing *agent* interacts with, comprising everything outside the agent, is called *environment*
- A complete specification of an environment defines a *task*, one instance of reinforcement learning problem

The interaction happens continually at each of a time sequence $t=0,1,2,3……$

At each $t$, agent receives a state $S_t$, on that basis select an action $A_t \in \mathbb{A}(S_t)$, one time step later, agent receives reward $R_{t+1}$, and the environment goes into $S_{t+1}$. Before loop goes recurrently.

<img src="../../Reinforcement Learning/Reinforcement Learning Notes.assets/image-20200521162257021.png" alt="image-20200521162257021" style="zoom:40%;" />

**<font color='steelblue'>The agent implements a mapping from states to probabilities of selecting the each possible action.</font>**

- For example, $\pi_{t}(a|S)$ represents the prob. of choosing action a when in state S of time-step t.



**<font color='deepskyblue'>Sometimes the boundary between agent and environment can be confusing.</font>**

- General rule we follow is that anything cannot be changed arbitrarily by the agent is considered to be outside of it and thus part of the environment.



**<font color='yellowgreen'>Any problem of learning goal-directed behavior
can be reduced to three signals passing back and forth between an agent and
its environment</font>**

- **Signal1**: represent the choices made by the agent. (actions)
- **Signal2**: represent the basis where choices are made. (states)
- **Signal3**: define the agent’s goal. (rewards)

This frame is not sufficient to represent all, but widely useful.



**<font color='olive'>3.2 Goals and Rewards</font>**

- The rewards are computed in th environment rather than in the agent.
- The reason we do this is the ultimate goal of agent should be something over which it has imperfect controls.



**<font color='olive'>3.3 Returns</font>**

Denote the sequence of rewards after time $t$ to be $R_{t+1}, R_{t+2}…$, in general, we seek to maximize the expected return. A straightforward way to express is:

$G_{t} = R_{t+1}+R_{t+2}+R_{t+3}+…+R_T$

**Use above denotion to introduce 2 reinforcement learning tasks:**

- **<font color='cornflowerblue'>Episodic</font>**: such as plays of a game, trips through a maze, or any repeated interactions. Each episode ends in a special state called **Terminal State ($S_T$)**.

- **<font color='cornflowerblue'>Continuing</font>**: in many cases, agent-environment interactions cannot be break into episodes, but goes continually without limit. $T=\infty$.

- **<font color='royalblue'>Discounting</font>**: this is an additional concept other than **Episodic** and **Continuing**. Sometimes future rewards need to be discounted when setting up our current goal.

  $G_{t} = R_{t+1}+\gamma R_{t+1}+\gamma^{2}R_{t+2}+\gamma^{3}R_{t+3}+…=\sum_{k=0}^{\infty}(\gamma^{k}R_{t+k+1})$, where $\gamma$ is called discount rate and $0\le\gamma\le1$



**<font color='olive'>3.4 Unified Notations for Episodic and Continuing Tasks</font>**

We use $S_{t,i},A_{t,i},R_{t,i},\pi_{t,i},T_{i}$, to represent state at time $t$ of episode $i$.

Another notation is for covering both episodic and continuing tasks.

<img src="../../Reinforcement Learning/Reinforcement Learning Notes.assets/image-20200521171409571.png" alt="image-20200521171409571" style="zoom:40%;" />

The solid-squared state is called ***absorbing state*** for episodic task. This is corresponding to the end of the episode. Rewards of ***absorbing state*** are always 0.

By applying ***absorbing state***, we can unified write the current return of both episodic and continuing tasks in  the way of:

$G_{t}=\sum_{k=0}^{T-t-1}\gamma^{k}R_{t+k+1}$





**<font color='olive'>3.5 The Markov Property</font>**

- ***State Signal***:
  1. The state signal should include immediate sensations such as sensory measurement. But of course, sometimes more than that
  2. The state signal should not inform the agent of everything about the environment, or even everything that would be useful for making decisions
  3. What we would expect, is a state signal that summarizes past sensations compactly, yet in such a way that all relevant information is retained. A state signal that succeeds in retaining all relevant information is said to be ***Markov***, or to have the ***Markov property***

**<font color='steelblue'>Consider how a general environment respond at time *t+1* given the action at time *t*.</font>**

$PR\{R_{t+1}=r, S_{t+1}=s’|S_0,A_0,R_1…S_{t-1},A_{t-1},R_{t},S_t,A_t\}$

**<font color='steelblue'>If the state signal has Markov Property, then the environment's response at t+1 depends only on the state and action representations at t</font>**

$p(s’,r|s,a)=PR\{R_{t+1}=r, S_{t+1}=s’|S_t,A_t\}$   (3.5)

***If an state has Markov property, then its one-step dynamics (3.5) enable us to predict the next state and expected next reward given the current state and action***

***By iterating this equation (3.5), one can also predict all future states and expected reward***

**<font color='yellowgreen'>For all of these reasons, it is useful to think of the state at each time step as an approximation to a Markov state, although one should remember that it may not fully satisfy the Markov property.</font>**





**<font color='olive'>3.6 Markov Decision Process</font>**

A reinforcement learning task that satises the Markov property is called a ***Markov decision process, or MDP.***

- if the state and action space is finite, we call this Finite ***Markov Decision Process, or Finite MDP***
- 90% reinforcement learning problems can be covered by Finite MDP.



**<font color='mediumseagreen'>A particular nite MDP is dened by its state and action sets and by the one-step dynamics of the environment. Given any state and action s and a, the probability of each possible pair of next state and reward, s' r is denoted</font>**

$p(s’,r|s,a) = Pr\{S_{t+1}=s’,R_{t+1}=r|S_{t}=s,A_{t}=a\}$   (3.6)

Given the dynamic state specified by (3.6), one can compute any anything else of the environment, such as the 

- **expected rewards of state-action pairs.**

  $r(s,a)=\mathbb{E}[R_{t+1}|S_t=s,A_t=a]=\sum_{r\in \mathcal{R}}rp(s,a)=\sum_{r\in \mathcal{R}}r\sum_{s'\in \mathcal{S}}p(s',r|s,a)$, where $\mathcal{R}$ and $\mathcal{S}$ are reward and state spaces.

- **State-Transition Probabilities**

  $p(s’|s,a)=Pr\{S_{t+1}=s’|S_t=s,A_t=a\}=\sum_{r\in R}p(s’,r|s,a)$

- **Expected Rewards for state-action-next-state triples**

  $r(s’,a,s)=\mathbb{E}[R_{t+1}|S_{t+1}=s’,S_t=s,A_t=a]= \sum_{r\in \mathcal{R}}rp(r|s’,a,s) =\frac{\sum_{r\in R} p(s’,r|s,a)}{p(s’|s,a)}$






**<font color='olive'>3.7 Value Functions</font>**

Recall that a policy $\pi$, is a mapping from each state $s\in \mathcal{S}$, and each action $a \in \mathcal{A}(s)$, to the probability $\pi(a|s)$ of taking action a in state s.

- ***<u>Informally, the value of a state $s$ under a policy $\pi$, denoted $v_{\pi}(s)$, is the expected return when starting in $s$ and following $\pi$ thereafter.</u>***

  For MDPs, we can define $v_{\pi}(s)$ formally as:

  $v_{\pi}(s)=\mathbb{E}_{\pi}[G_t|S_t=s]=\mathbb{E}_{\pi}[\sum_{k=0}^{\infty}\gamma^{k}R_{t+k+1}|S_t=s]$

  where $\mathbb{E}[.]$ denotes the expected value of a random variable given that agent always follow policy $\pi$.

  ***<font color='steelblue'> We call function</font>*** $v_{\pi}$ ***<font color='steelblue'>the state-value function for policy</font>*** $\pi$.

- <u>***Similarly, we define the value of taking action $a$ in state $s$ under a policy $\pi$, denoted $q_{\pi}(s,a)$, as the expected return starting from $s$, taking the action a, and thereafter following policy $\pi$.***</u>

  $q_{\pi}(s,a) = \mathbb{E}_{\pi}[G_t|S_t=s,A_t=a]=\mathbb{E}_{\pi}[\sum_{k=0}^{\infty}\gamma^{k}R_{t+k+1}|S_t=s,A_t=a]$

  ***<font color='steelblue'> We call function</font>*** $q_{\pi}$ ***<font color='steelblue'>the action-value function for policy</font>*** $\pi$.

- Value functions $v_{\pi}$ and $q_{\pi}$ can be estimated by experience:

  - if an agent follows policy $\pi$ in each state $s$ encountered, and maintain the average of the reward followed in that state, then it will finanlly converge to $v_{\pi}$ as approach to infinity:

    $lim_{N\rightarrow\infty}\{\frac{1}{N}\sum_{t=0}^{N}r_{t}(s,\pi(s))\}\rightarrow v_{\pi}(s)$

  - if seperate average are recorded for each action taken (mapped by the policy), then the average will finally converge to $q_{\pi}$

    $lim_{m_{1}\rightarrow\infty}\{\frac{1}{m_{1}}\sum^{m_{1}}r(s,a_{1}=\pi(s))\}\rightarrow q_{\pi}(s,a_{1})$

    $lim_{m_{2}\rightarrow\infty}\{\frac{1}{m_{2}}\sum^{m_{2}}r(s,a_{2}=\pi(s))\}\rightarrow q_{\pi}(s,a_{2})$

    ...

    $lim_{m_{n}\rightarrow\infty}\{\frac{1}{m_{n}}\sum^{m_{n}}r(s,a_{n}=\pi(s))\}\rightarrow q_{\pi}(s,a_{n})$

    where $\mathcal{A}(s)=\{a_{1},a_{2},…,a_{n}\}$ and $\sum_{i=0}m_{i}=N$

  - The estimation method **Monte Carlo Simulation** will be introduced later.

- A fundamental property of value functions used throughout reinforcement learning and dynamic programming is that **<u>they satisfy particular recursive relationships.</u>** For any policy $\pi$ and any states $s$, the following consistency condition holds between the value of $s$ and the value of its possible successor states:

  $v_{\pi}(s)=\mathbb{E}[G_t|S_t=s]$

  $=\mathbb{E}_{\pi}[\sum_{k=0}^{\infty}\gamma^{k}R_{t+k+1}|S_t=s]$

  $=\mathbb{E}_{\pi}[R_{t+1}+\sum_{k=1}^{\infty}\gamma^{k}R_{t+k+1}|S_t=s]$

  $=\mathbb{E}_{\pi}[R_{t+1}|S_t=s]+\mathbb{E}_{\pi}[\gamma\sum_{k=0}^{\infty}\gamma^{k}R_{t+k+2}|S_t=s]$

  $=\sum_{r\in R}r*p(r|s)+\mathbb{E}_{\pi}[\gamma\sum_{k=0}^{\infty}\gamma^{k}R_{t+k+2}|S_t=s]$

  $=\sum_{r\in R}r*\sum_{a \in \mathcal{A}(s)}\pi(a|s)p(r|s,a)+\gamma\mathbb{E}_{\pi}[\sum_{k=0}^{\infty}\gamma^{k}R_{t+k+2}|S_t=s]$

  $=\sum_{r\in R}r*\sum_{a \in \mathcal{A}(s)}\pi(a|s)\sum_{s'\in \mathcal{S}}p(s’,r|s,a)+\gamma\mathbb{E}_{\pi}[\sum_{k=0}^{\infty}\gamma^{k}R_{t+k+2}|S_t=s]$

  $=\sum_{a \in \mathcal{A}(s)}\pi(a|s)\sum_{s'\in \mathcal{S},r\in \mathcal{R}}p(s’,r|s,a)(r+\gamma\mathbb{E}_{\pi}[\sum_{k=0}^{\infty}\gamma^{k}R_{t+k+2}|S_{t+1}=s'])$

  $=\sum_{a}\pi(a|s)\sum_{s’,r}p(s’,r|s,a)(r+\gamma v_{\pi}(s'))$      （3.12）

  This equation is called ***Bellman Equation*** for $v_{\pi}(s)$, it expresses a relationship between the value of a state and the values of its successor states.

- Think of looking ahead from one state to its possible successor states:

  <img src="../../Reinforcement Learning/Reinforcement Learning Notes.assets/image-20200525134131000.png" alt="image-20200525134131000" style="zoom:33%;" />

  *open circle stands for state, solid circle stands for state-action pair*

  **<font color='cornflowerblue'>So the Bellman Equation states that the value of the start state must equal the discounted value of the expected next state, plus the expected reward along the way.</font>**



**<font color='olive'>3.8 Optimal Value Functions</font>**

***<font color='darkolivegreen'>Value functions define a partial ordering over policies</font>***

- A policy $\pi$ is defined to be better than or equal to policy $\pi’$ if its expected return is greater than or equal to that of $\pi’$ for all states.

- In math language: $\pi \ge \pi’$ if and only if $v_{\pi}(s)\ge v_{\pi’}(s)$ for all $s\in \mathcal{S}$

- <u>***There is always at least one policy better than or equal to all other policies, this is called <font color='steelblue'>optimal policy</font>, denoted as***</u> $\pi_{*}$. <u>***They share the same <font color='steelblue'>optimal state-value function</font>***</u> $v_{*}$.

  $v_{*}(s)=max_{\pi}v_{\pi}(s)$ for all $s \in \mathcal{S}$

- <u>***Optimal policies also share the same <font color='steelblue'>optimal action-value function</font>***</u> $q_{*}$.

  $q_{*}(s,a) = max_{\pi}q_{\pi}(s,a)$ for all $s \in \mathcal{S}$ and $a \in \mathcal{A}(s)$.

- For the state-action pair (s,a), $q_{*}(s,a)$ gives the expected return for taking action a in the state s, and follow the optimal policy thereafter.

  $q_{*}(s,a) = \mathbb{E}[R_{t+1}+\gamma v_{*}(S_{t+1})|S_t=s,A_t=a]$

***<font color='darkolivegreen'>Bellman Optimality Equation</font>***

- **<font color='deepskyblue'>The value of a state under an optimal policy must equal the expected return for the best action from that state</font>**

-  ***Bellman Optimality Equation*** for $v_{*}$  (3.16 and 3.17)

  $v_{*}(s) = max_{a\in \mathcal{A}(s)}q_{\pi_{*}}(s,a)$

  $=max_{a\in \mathcal{A}(s)}\mathbb{E}_{\pi_{*}}[G_t|S_t=s,A_t=a]$

  $=max_{a\in \mathcal{A}(s)}\mathbb{E}_{\pi_{*}}[\sum_{k=0}\gamma^{k}R_{t+k+1}|S_t=s,A_t=a]$

  $=max_{a\in \mathcal{A}(s)}\mathbb{E}_{\pi_{*}}[R_{t+1}+\gamma\sum_{k=0}\gamma^{k}R_{t+k+2}|S_t=s,A_t=a]$

  $=max_{a\in \mathcal{A}(s)}\mathbb{E}_{\pi_{*}}[R_{t+1}+\gamma v_{*}(S_{t+1})|S_t=s,A_t=a]$    （3.16）

  $=max_{a\in \mathcal{A}(s)}\sum_{s’,r}p(s’,r|s,a)(r+\gamma v_{*}(s'))$                  （3.17）

-  ***Bellman Optimality Equation*** for $q_{*}$

  $q_{*}(s,a) = \mathbb{E}_{\pi_{*}}[R_{t+1}+\gamma \mathcal{max}_{a’}q_{*}(S_{t+1},a’)|S_t=s,A_t=a]$

  $