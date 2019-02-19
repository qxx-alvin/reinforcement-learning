#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
using namespace std;

static int TOTAL_RUNS = 1000;
static int STEPS_PER_RUN = 200000;
static int NUM_ACTIONS = 10;
static bool NON_STATIONARY = true;

static int STEPS_TO_AVERAGE = 100000;

void e_greedy_learning_curve(double epsilon)
{
	srand((unsigned)time(0));
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(0.0, 1.0);
	std::normal_distribution<double> distribution_delta(0.0, 0.01);

	vector<double> sum_rew_over_runs(STEPS_PER_RUN, 0);
	for (int i = 0; i < TOTAL_RUNS; i++)
	{
		if (i % 100 == 0)
			printf("Run %d\n", i);

		vector<double> Q(NUM_ACTIONS, 0);
		vector<int> N(NUM_ACTIONS, 0);
		vector<double> q(NUM_ACTIONS, 0);
		if (!NON_STATIONARY)
		{
			for (int j = 0; j < NUM_ACTIONS; j++)
				q[j] = distribution(generator);
		}

		for (int j = 0; j < STEPS_PER_RUN; j++)
		{
			int act;
			double r = rand() / (RAND_MAX + 1.0);
			if (r < epsilon)
			{
				// randomly select an action
				act = rand() / (RAND_MAX + 1.0) * NUM_ACTIONS;
			}
			else
			{
				// select action with largest advantage
				auto maxPosition = max_element(Q.begin(), Q.end());
				act = maxPosition - Q.begin();
			}

			// random walk
			if (NON_STATIONARY)
			{
				for (int k = 0; k < NUM_ACTIONS; k++)
					q[k] += distribution_delta(generator);
			}

			// get reward
			std::normal_distribution<double> distribution_reward(q[act], 1);
			double reward = distribution_reward(generator);

			// update estimates
			N[act] += 1;
			Q[act] = Q[act] + 1.0 / N[act] * (reward - Q[act]);

			// sum reward across runs
			sum_rew_over_runs[j] += reward;
		}
	}

	// save to file
	FILE *fp = NULL;
	fopen_s(&fp, "e-greedy_learning_curve.txt", "w");
	if (fp == NULL)
		return;

	for (int i = 0; i < STEPS_PER_RUN; i++)
		fprintf(fp, "%f ", sum_rew_over_runs[i] / TOTAL_RUNS);
	fclose(fp);
}

void e_greedy_parameter_study()
{
	srand((unsigned)time(0));
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(0.0, 1.0);
	std::normal_distribution<double> distribution_delta(0.0, 0.01);

	vector<double> av_reward_per_setting;
	vector<double> epsilon;
	for (int i = -8; i <= -2; i++)
	{
		epsilon.push_back(pow(2, i));
		//printf("%f\n", coef[coef.size() -1]);
	}

	for (int n = 0; n < epsilon.size(); n++)
	{
		double ep = epsilon[n];
		printf("epsilon = %f\n", ep);

		vector<double> av_rew_per_run;
		for (int i = 0; i < TOTAL_RUNS; i++)
		{
			if (i % 100 == 0)
				printf("-Run %d\n", i);

			vector<double> Q(NUM_ACTIONS, 0);
			vector<int> N(NUM_ACTIONS, 0);
			vector<double> q(NUM_ACTIONS, 0);
			if (!NON_STATIONARY)
			{
				for (int j = 0; j < NUM_ACTIONS; j++)
					q[j] = distribution(generator);
			}
			double av_rew = 0;
			int cnt = 0;

			for (int j = 0; j < STEPS_PER_RUN; j++)
			{
				int act;
				double r = rand() / (RAND_MAX + 1.0);
				if (r < ep)
				{
					// randomly select an action
					act = rand() / (RAND_MAX + 1.0) * NUM_ACTIONS;
				}
				else
				{
					// select action with largest advantage
					auto maxPosition = max_element(Q.begin(), Q.end());
					act = maxPosition - Q.begin();
				}

				// random walk
				if (NON_STATIONARY)
				{
					for (int k = 0; k < NUM_ACTIONS; k++)
						q[k] += distribution_delta(generator);
				}

				// get reward
				std::normal_distribution<double> distribution_reward(q[act], 1);
				double reward = distribution_reward(generator);

				// update estimates
				N[act] += 1;
				Q[act] = Q[act] + 1.0 / N[act] * (reward - Q[act]);

				// sum reward
				if (j > STEPS_TO_AVERAGE)
				{
					cnt++;
					av_rew = av_rew + 1.0 / cnt * (reward - av_rew);
				}
			}
			av_rew_per_run.push_back(av_rew);
		}
		double sum = std::accumulate(std::begin(av_rew_per_run), std::end(av_rew_per_run), 0.0);
		av_reward_per_setting.push_back(sum / av_rew_per_run.size());
	}


	// save to file
	FILE *fp = NULL;
	fopen_s(&fp, "e_greedy_parameter_study.txt", "w");
	if (fp == NULL)
		return;

	for (int i = 0; i < av_reward_per_setting.size(); i++)
		fprintf(fp, "%f ", av_reward_per_setting[i]);
	fclose(fp);
}



