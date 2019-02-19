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

void gradient_learning_curve(double alpha)
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

		vector<double> H(NUM_ACTIONS, 0);
		vector<double> pi(NUM_ACTIONS, 0);		
		vector<double> q(NUM_ACTIONS, 0);
		if (!NON_STATIONARY)
		{
			for (int j = 0; j < NUM_ACTIONS; j++)
				q[j] = distribution(generator);
		}
		double av_rew = 0;

		for (int j = 0; j < STEPS_PER_RUN; j++)
		{
			int act;
			// choose action according to probability
			double sum_exp = 0;
			for (int k = 0; k < NUM_ACTIONS; k++)
				sum_exp += exp(H[k]);

			for (int k = 0; k < NUM_ACTIONS; k++)
				pi[k] = exp(H[k]) / sum_exp;

			double r = rand() / (RAND_MAX + 1.0);
			double cum_pi = pi[0];		// 累积概率
			for (int k = 0; k < NUM_ACTIONS; k++)
			{
				if (r < cum_pi)
				{
					act = k;
					break;
				}
				else
					cum_pi += pi[k + 1];
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
			av_rew = av_rew + 1.0 / (j + 1)*(reward - av_rew);

			// update preference H
			for (int k = 0; k < NUM_ACTIONS; k++)
			{
				if (k == act)
					H[k] += alpha*(reward - av_rew) * (1 - pi[k]);
				else
					H[k] -= alpha*(reward - av_rew) * pi[k];
			}

			// sum reward across runs
			sum_rew_over_runs[j] += reward;
		}
	}

	// save to file
	FILE *fp = NULL;
	fopen_s(&fp, "gradient_learning_curve.txt", "w");
	if (fp == NULL)
		return;

	for (int i = 0; i < STEPS_PER_RUN; i++)
		fprintf(fp, "%f ", sum_rew_over_runs[i] / TOTAL_RUNS);
	fclose(fp);
}

void gradient_parameter_study()
{
	srand((unsigned)time(0));
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(0.0, 1.0);
	std::normal_distribution<double> distribution_delta(0.0, 0.01);

	vector<double> av_reward_per_setting;
	vector<double> alpha;
	for (int i = -8; i <= -4; i++)
	{
		alpha.push_back(pow(2, i));
		//printf("%f\n", coef[coef.size() -1]);
	}

	for (int n = 0; n < alpha.size(); n++)
	{
		double alp = alpha[n];
		printf("alpha = %f\n", alp);

		vector<double> av_rew_per_run;
		for (int i = 0; i < TOTAL_RUNS; i++)
		{
			if (i % 100 == 0)
				printf("-Run %d\n", i);

			vector<double> H(NUM_ACTIONS, 0);
			vector<double> pi(NUM_ACTIONS, 0);
			vector<double> q(NUM_ACTIONS, 0);
			if (!NON_STATIONARY)
			{
				for (int j = 0; j < NUM_ACTIONS; j++)
					q[j] = distribution(generator);
			}
			double av_rew = 0;
			double av_rew_last_steps = 0;
			int cnt = 0;

			for (int j = 0; j < STEPS_PER_RUN; j++)
			{
				int act;
				// choose action according to probability
				double sum_exp = 0;
				for (int k = 0; k < NUM_ACTIONS; k++)
					sum_exp += exp(H[k]);

				for (int k = 0; k < NUM_ACTIONS; k++)
					pi[k] = exp(H[k]) / sum_exp;

				double r = rand() / (RAND_MAX + 1.0);
				double cum_pi = pi[0];		// 累积概率
				for (int k = 0; k < NUM_ACTIONS; k++)
				{
					if (r < cum_pi)
					{
						act = k;
						break;
					}
					else
						cum_pi += pi[k + 1];
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
				av_rew = av_rew + 1.0 / (j + 1)*(reward - av_rew);

				// update preference H
				for (int k = 0; k < NUM_ACTIONS; k++)
				{
					if (k == act)
						H[k] += alp*(reward - av_rew) * (1 - pi[k]);
					else
						H[k] -= alp*(reward - av_rew) * pi[k];
				}

				// sum reward
				if (j > STEPS_TO_AVERAGE)
				{
					cnt++;
					av_rew_last_steps = av_rew_last_steps + 1.0 / cnt * (reward - av_rew_last_steps);
				}
			}
			av_rew_per_run.push_back(av_rew_last_steps);
		}
		double sum = std::accumulate(std::begin(av_rew_per_run), std::end(av_rew_per_run), 0.0);
		av_reward_per_setting.push_back(sum / av_rew_per_run.size());
	}


	// save to file
	FILE *fp = NULL;
	fopen_s(&fp, "gradient_parameter_study.txt", "w");
	if (fp == NULL)
		return;

	for (int i = 0; i < av_reward_per_setting.size(); i++)
		fprintf(fp, "%f ", av_reward_per_setting[i]);
	fclose(fp);
}



