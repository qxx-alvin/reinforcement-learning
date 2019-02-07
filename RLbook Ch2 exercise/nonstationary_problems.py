###############
# demonstrate the difficulties that sample-average methods have for nonstationary problems
###############

import matplotlib.pyplot as plt
import numpy as np

# 10-armed bandit
epsilon = 0.1
total_runs = 2000
steps_per_run = 10000
constant_step = 0.1
actions = 10

q = []          # true action value for all actions
Q_saav = []     # action value estimates using sample-average
Q_cons = []     # action value estimates using constant step-size
N_saav = []     # number of each action visited
N_cons = []
# rew_saav = []     # rewards for sample-average
# rew_cons = []       # rewards for constant step-size
sum_rew_saav = [0] * steps_per_run
sum_rew_cons = [0] * steps_per_run
optimal_visit_saav = [0] * steps_per_run
optimal_visit_cons = [0] * steps_per_run

for n in range(total_runs):
    if (n % 1 == 0):
        print("Run %d" % n)

    q.clear()
    Q_saav.clear()
    Q_cons.clear()
    N_saav.clear()
    N_cons.clear()
    # rew_saav.clear()
    # rew_cons.clear()

    for i in range(actions):
        q.append(0)
        Q_saav.append(0)
        Q_cons.append(0)
        N_saav.append(0)
        N_cons.append(0)

    for i in range(steps_per_run):
        # epsilon-greedy
        r = np.random.rand()
        if r < epsilon:
            act_saav = np.random.randint(actions)
            act_cons = np.random.randint(actions)
        else:
            act_max = [j for j in range(actions) if Q_saav[j] == np.max(Q_saav)]
            act_saav = act_max[np.random.randint(act_max.__len__())]
            act_max = [j for j in range(actions) if Q_cons[j] == np.max(Q_cons)]
            act_cons = act_max[np.random.randint(act_max.__len__())]

        # whether optimal is visited
        optimal_visit_saav[i] += (act_saav == np.argmax(q))
        optimal_visit_cons[i] += (act_cons == np.argmax(q))

        # random walk
        for j in range(actions):
            q[j] += np.random.normal(0, 0.01)

        # get reward
        reward_saav = np.random.normal(q[act_saav], 1)
        reward_cons = np.random.normal(q[act_cons], 1)
        # rew_saav.append(reward_saav)
        # rew_cons.append(reward_cons)

        # update estimates
        N_saav[act_saav] += 1
        N_cons[act_cons] += 1
        Q_saav[act_saav] = Q_saav[act_saav] + 1.0 / N_saav[act_saav] * (reward_saav - Q_saav[act_saav])
        Q_cons[act_cons] = Q_cons[act_cons] + constant_step * (reward_cons - Q_cons[act_cons])

        # sum reward across runs
        sum_rew_saav[i] += reward_saav
        sum_rew_cons[i] += reward_cons


# plot
plt.figure()
plt.plot(list(range(steps_per_run)), np.array(sum_rew_saav) / total_runs, 'r', label='sample average')
plt.plot(list(range(steps_per_run)), np.array(sum_rew_cons) / total_runs, 'b', label='constant step-size')
plt.xlabel(u'Steps')
plt.ylabel(u"Average reward")
plt.legend()
plt.show()

plt.figure()
plt.plot(list(range(steps_per_run)), np.array(optimal_visit_saav) / total_runs, 'r', label='sample average')
plt.plot(list(range(steps_per_run)), np.array(optimal_visit_cons) / total_runs, 'b', label='constant step-size')
plt.xlabel(u'Steps')
plt.ylabel(u"Optimal Action")        
plt.legend()
plt.show()      
        
        
        
        
        
        
        
        
        
        
        