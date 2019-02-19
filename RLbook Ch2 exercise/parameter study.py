###############
# parameter study for various methods for k-bandit problem
# stationary
###############

import matplotlib.pyplot as plt
import numpy as np

# 10-armed bandit
total_runs = 1000
steps_per_run = 20000
steps_to_average = 10000
actions = 10

# e-greedy
epsilon = [2**(-i) for i in range(7, 1, -1)]

# e-greedy constant step-size
constant_step = 0.1

# q = []          # true action value for all actions
# Q_saav = []     # action value estimates using sample-average
# Q_cons = []     # action value estimates using constant step-size
# N_saav = []     # number of each action visited
# N_cons = []
# rew_saav = []     # rewards for sample-average
# rew_cons = []       # rewards for constant step-size
# sum_rew_saav = [0] * steps_per_run
# sum_rew_cons = [0] * steps_per_run
# optimal_visit_saav = [0] * steps_per_run
# optimal_visit_cons = [0] * steps_per_run

av_rew_buf = []
# for i in range(actions):
#     q.append(np.random.normal(0, 1))

for ep in epsilon:
    print("Epsilon %f" % ep)

    # q.clear()
    # Q_saav.clear()
    # Q_cons.clear()
    # N_saav.clear()
    # N_cons.clear()
    # rew_saav.clear()
    # rew_cons.clear()

    # for i in range(actions):
    #     # q.append(0)
    #     # Q_saav.append(0)
    #     # Q_cons.append(0)
    #     # N_saav.append(0)
    #     # N_cons.append(0)
    #     Q.append(0)
    #     N.append(0)
    av_rew_per_run = []
    for n in range(total_runs):
        if (n % 10 == 0):
            print("-Run %d" % n)

        Q = [0] * actions
        N = [0] * actions
        av_rew = 0
        j = 0
        q = [0] * actions

        for i in range(steps_per_run):
            # if i % 10000 == 0 and i != 0:
            #     print("---Steps %d" % i)
            # epsilon-greedy
            r = np.random.rand()
            if r < ep:
                act = np.random.randint(actions)
            else:
                act_max = [j for j in range(actions) if Q[j] == np.max(Q)]
                act = act_max[np.random.randint(act_max.__len__())]
                
            # whether optimal is visited
            # optimal_visit_saav[i] += (act_saav == np.argmax(q))
            # optimal_visit_cons[i] += (act_cons == np.argmax(q))

            # random walk
            for j in range(actions):
                q[j] += np.random.normal(0, 0.01)

            # get reward
            reward = np.random.normal(q[act], 1)
            # rew_saav.append(reward_saav)
            # rew_cons.append(reward_cons)

            # update estimates
            N[act] += 1
            Q[act] = Q[act] + 1.0 / N[act] * (reward - Q[act])
            # Q_cons[act_cons] = Q_cons[act_cons] + constant_step * (reward_cons - Q_cons[act_cons])

            # sum reward across runs
            # sum_rew_saav[i] += reward_saav
            # sum_rew_cons[i] += reward_cons

            if i > steps_to_average:
                j += 1
                av_rew = av_rew + 1.0 / j * (reward - av_rew)
        av_rew_per_run.append(av_rew)

    av_rew_buf.append(np.mean(av_rew_per_run))

# save to file
fp = open("e-greedy.txt", 'w')
fp.write(' '.join([str(x) for x in av_rew_buf]))
fp.close()

# plot
# plt.figure()
# plt.plot(list(range(steps_per_run)), np.array(sum_rew_saav) / total_runs, 'r')
# # plt.plot(list(range(steps_per_run)), np.array(sum_rew_cons) / total_runs, 'b', label='constant step-size')
# plt.xlabel(u'')
# plt.ylabel(u"Average reward over last 100,000 steps")
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(list(range(steps_per_run)), np.array(optimal_visit_saav) / total_runs, 'r', label='sample average')
# plt.plot(list(range(steps_per_run)), np.array(optimal_visit_cons) / total_runs, 'b', label='constant step-size')
# plt.xlabel(u'Steps')
# plt.ylabel(u"Optimal Action")        
# plt.legend()
# plt.show()      
        
        
        
        
        
        
        
        
        
        
        