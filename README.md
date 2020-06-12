# Reinforcement Learning Resources
[Lectures, Books, Surveys and Thesis of Reinforcement Learning](https://github.com/aikorea/awesome-rl)

[An Outsider’s Tour of Reinforcement Learning](http://www.argmin.net/2018/06/25/outsider-rl/)

[Reinforcement Learning](https://github.com/RL-Research-Cohiba/Reinforcement_Learning)

[强化学习从入门到放弃](https://github.com/wwxFromTju/awesome-reinforcement-learning-zh)

[OpenAI DeepRL Courses](https://spinningup.openai.com/en/latest/)

[Dynamic Programming Problems](https://blog.usejournal.com/top-50-dynamic-programming-practice-problems-4208fed71aa3)

# Study Notes and Codes

Now working on Chapter5 Monte Carlo Method of *Reinforcement Learning: An Introduction*

<a download="Reinforcement Learning Notes.md" href='https://github.com/leafsigh/Reinforcement_Learning/blob/master/Reinforcement-Learning-Notes.ipynb'>Study Notes</a>
- Images and colored texts cannot show correctly on Github. Please copy the above link into [nbviewer](https://nbviewer.jupyter.org) to get a correct view.

<a download="N_armed bandit.ipynb" href='https://github.com/leafsigh/Reinforcement_Learning/blob/master/N_armed%20bandit.ipynb'>N-Armed-Bandit Code</a>

<a download="GridWorld_DP.ipynb" href='https://github.com/leafsigh/Reinforcement_Learning/blob/master/GridWorld_DP.ipynb'>NxN GridWorld Code (Only contain one-step policy evaluation)</a>

<a download="GridWorld_by_PolicyIteration.ipynb" href='https://github.com/leafsigh/Reinforcement_Learning/blob/master/GridWorld_by_PolicyIteration.ipynb'>NxN GridWorld (by Policy Iteration)</a>

<a download="CarRental_PolicyIteration.ipynb" href='https://github.com/leafsigh/Reinforcement_Learning/blob/master/Policy_Iteration-Car_Rental_Problem%20.ipynb'>CarRental Policy Iteration (still working on it)</a>

<a download="BlackJack_MonteCarlo.ipynb" href='https://github.com/leafsigh/Reinforcement_Learning/blob/master/BlackJack_by_MonteCarlo.ipynb'>BlackJack by Monte Carlo (has finished policy evaluation part)</a>

# Study Note
- The study note of *Reinforcement Learning: An Introduction*. Contents in .md and .ipynb are the same.

# N-Armed-Bandit Problem
N-Armed-Bandit.ipynb now has included the entire algorithms of this interesting problem.
- 4 action selelcting algorithms: epsilon-greedy, softmax, upper bound confidence (UCB) and gradient ascent (preference estimation).
- 2 data generation methods: stationary and nonstationary.
- 2 initial value setup methods: add baseline and setup burning period.

Future works on this script will focus on optimizing the performance and correcting potential bugs.
Solutions by `Gym` will be added later.

# GridWorld Problem
- Size of the GridWorld can be changed at will. To get the same result as *Reinforcement Learning: An Introduction*, change n=4.
- The GridWorld_DP.ipynb only contains the **policy evaluation**.
- The GridWorld_by_PolicyIteration.ipynb contains completer **policy iteration** procedure. **Value iteration** is a special case in **policy iteration**, which can be adapted by the code.

Future works will focus on optimizing the efficiency and adding visualization.
Solutions by `Gym` will be added later.

# BlackJack
- The BlackJack problem is soluted by **Monte Carlo Method**.
- Has finished the policy evaluation part. Currently working on policy improvement.
- Solutions by `Gym` will be added later.
![](https://github.com/leafsigh/Reinforcement-Learning/blob/master/BlackJack_Visualization/png_to_gif.gif)
