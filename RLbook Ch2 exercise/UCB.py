###############
# UCB
###############

import matplotlib.pyplot as plt
import numpy as np

# 10-armed bandit
total_runs = 100
steps_per_run = 1000
c = 2
actions = 10

non_stationary = False

# rew_saav = []     # rewards for sample-average
# rew_cons = []       # rewards for constant step-size
sum_rew = [0] * steps_per_run
# sum_rew_cons = [0] * steps_per_run
# optimal_visit_saav = [0] * steps_per_run
# optimal_visit_cons = [0] * steps_per_run

for n in range(total_runs):
    if (n % 10 == 0):
        print("Run %d" % n)

    Q = [0] * actions
    N = [0] * actions
    A = [0] * actions
    q = [0] * actions
    if non_stationary == False:
        for j in range(actions):
            q[j] = np.random.normal(0, 1)

    for i in range(steps_per_run):
        # choose actions that haven't been chosen
        act_null = [j for j in range(actions) if N[j] == 0]
        
        if (act_null.__len__() > 0):
            # randomly select an action
            act = act_null[np.random.randint(act_null.__len__())]
        else:
            # select action with largest advantage
            act_max = [j for j in range(actions) if A[j] == np.max(A)]
            act = act_max[np.random.randint(act_max.__len__())]       

        # random walk
        if non_stationary:
            for j in range(actions):
                q[j] += np.random.normal(0, 0.01)

        # get reward
        reward = np.random.normal(q[act], 1)

        # update estimates
        N[act] += 1
        Q[act] = Q[act] + 1.0 / N[act] * (reward - Q[act])

        # update advantages
        for j in range(actions):
            if N[j] != 0:
                A[j] = Q[j] + c * np.sqrt(np.log(i+1)/N[j])

        # sum reward across runs
        sum_rew[i] += reward

# plot
plt.figure()
plt.plot(list(range(steps_per_run)), np.array(sum_rew) / total_runs, 'r', label='UCB c = %d' % c)
# plt.plot(list(range(steps_per_run)), np.array(sum_rew_cons) / total_runs, 'b', label='constant step-size')
plt.xlabel(u'Steps')
plt.ylabel(u"Average reward")
plt.legend()
plt.show()

# plt.figure()
# plt.plot(list(range(steps_per_run)), np.array(optimal_visit_saav) / total_runs, 'r', label='sample average')
# plt.plot(list(range(steps_per_run)), np.array(optimal_visit_cons) / total_runs, 'b', label='constant step-size')
# plt.xlabel(u'Steps')
# plt.ylabel(u"Optimal Action")        
# plt.legend()
# plt.show()      
        
        
        
        
        
        
        
        
        
        
        