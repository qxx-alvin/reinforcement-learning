import matplotlib.pyplot as plt
import numpy as np

####### e-greedy #######
epsilon = list(range(-8, -1, 1))
fp = open('D:\\Program Files (x86)\\Microsoft Visual Studio 12.0\\my project\\Parameter study\\Parameter study\\e_greedy_parameter_study.txt', 'r')
e_greedy_data = fp.read().strip()
fp.close()
e_greedy_data = list(map(float, e_greedy_data.split(' ')))
# print(type(data), data)


####### UCB #######
c = list(range(3, 9, 1))
fp = open('D:\\Program Files (x86)\\Microsoft Visual Studio 12.0\\my project\\Parameter study\\Parameter study\\UCB_parameter_study.txt', 'r')
ucb_data = fp.read().strip()
fp.close()
ucb_data = list(map(float, ucb_data.split(' ')))

####### gradient #######
alpha = list(range(-8, -3, 1))
fp = open('D:\\Program Files (x86)\\Microsoft Visual Studio 12.0\\my project\\Parameter study\\Parameter study\\gradient_parameter_study.txt', 'r')
gradient_data = fp.read().strip()
fp.close()
gradient_data = list(map(float, gradient_data.split(' ')))


## PLOT ##
plt.figure()

plt.plot(epsilon, e_greedy_data, 'r')
plt.plot(c, ucb_data, 'b')
plt.plot(alpha, gradient_data, 'g')

plt.ylabel(u"Average reward over last 100000 steps")
# plt.ylim((1, 2))
# plt.xscale('logit')
x_tick = []#2**(-i) for i in range(-7, 9, 1)]
for i in range(-8,9,1):
    if (i < 0):
        x_tick.append("1/%d"%(2**(-i)))
    else:
        x_tick.append("%d"%(2**i))
plt.xticks(range(-8, 9, 1), x_tick)
plt.show()