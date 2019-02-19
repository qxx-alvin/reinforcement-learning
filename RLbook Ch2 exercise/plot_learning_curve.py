import matplotlib.pyplot as plt
import numpy as np

#### UCB
fp = open('D:\\Program Files (x86)\\Microsoft Visual Studio 12.0\\my project\\Parameter study\\Parameter study\\UCB_reward_vs_steps.txt', 'r')
ucb_data = fp.read()
fp.close()
ucb_data = ucb_data.strip().split(' ')
# print(float(data[data.__len__() - 1]))
ucb_data = list(map(float, ucb_data))
# print(type(data), data)

#### e-greedy
fp = open('D:\\Program Files (x86)\\Microsoft Visual Studio 12.0\\my project\\Parameter study\\Parameter study\\e-greedy_learning_curve.txt', 'r')
e_greedy_data = fp.read()
fp.close()
e_greedy_data = e_greedy_data.strip().split(' ')
# print(float(data[data.__len__() - 1]))
e_greedy_data = list(map(float, e_greedy_data))
# print(type(data), data)

#### gradient
fp = open('D:\\Program Files (x86)\\Microsoft Visual Studio 12.0\\my project\\Parameter study\\Parameter study\\gradient_learning_curve.txt', 'r')
gradient_data = fp.read()
fp.close()
gradient_data = gradient_data.strip().split(' ')
# print(float(data[data.__len__() - 1]))
gradient_data = list(map(float, gradient_data))
# print(type(data), data)


#### PLOT
plt.figure()
plt.plot(range(e_greedy_data.__len__()), e_greedy_data, 'r', label='e-greedy')
plt.plot(range(ucb_data.__len__()), ucb_data, 'b', label='UCB')
plt.plot(range(gradient_data.__len__()), gradient_data, 'g', label='gradient')
# # plt.plot(list(range(steps_per_run)), np.array(sum_rew_cons) / total_runs, 'b', label='constant step-size')
plt.xlabel(u'Steps')
plt.ylabel(u"Average reward")
# plt.ylim((1, 2))
plt.legend()
plt.show()