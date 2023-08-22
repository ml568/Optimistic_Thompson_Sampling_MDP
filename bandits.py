# -*- coding: utf-8 -*-
"""
# Code for Optimistic Thompson's Sampling for Bandit
"""
import os
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from tqdm import tqdm
from argparse import ArgumentParser
from multiprocessing import Pool

"""# Environment / Util"""

class Bandits():
    def __init__(self, mean_r):
        self.K = len(mean_r)
        self.R = mean_r
        self.total_reward = 0  # Cumulative reward within one experiment

        # Observation statistics
        self.emperical_rewards = np.zeros(self.K)
        self.observations = np.ones(self.K)

    def step(self, a):
        r = np.random.binomial(n=1, p= self.R[a])
        self.total_reward = self.total_reward + r
        self.emperical_rewards[a] += r
        self.observations[a] += 1

    def init(self):
        for i in range(self.K):
            self.step(i)


def running_avg(r):
    window = 10
    average_r = []
    for ind in range(len(r) - window + 1):
        average_r.append(np.mean(r[ind:ind + window]))
    for ind in range(window - 1):
        average_r.insert(0, np.nan)
    return average_r


def cum_me(r):
    return np.cumsum(r) / range(1, len(r) + 1)


"""# Algorithms
"""

def run_rand(m,epLen):
    r = []
    for i in range(epLen):
        arm = np.random.randint(m.K)
        m.step(arm)
        r.append(m.total_reward)
    return r

def run_best(m, best_arm, epLen):
    r = []
    for i in range(epLen):
        m.step(best_arm)
        r.append(m.total_reward)
    return r

def UCB(m, epLen):
    r = []
    m.init()
    for i in range(1, epLen+1):
        mu_hat = m.emperical_rewards /m.observations
        bonus = np.sqrt(2 * np.log(i)/m.observations)
        ucb = mu_hat + bonus
        arm = np.argmax(ucb)
        m.step(arm)
        r.append(m.total_reward)
    return r

def TS(m, epLen):
    r = []
    m.init()
    for i in range(1, epLen+1):
        ts = []
        for k in range(m.K):
            mu_hat = m.emperical_rewards[k] / m.observations[k]
            sigma = 1/m.observations[k]
            ts.append(np.random.normal(mu_hat, np.sqrt(sigma)))
        arm = np.argmax(ts)
        m.step(arm)
        r.append(m.total_reward)
    return r

def OTS(m, epLen):
    r = []
    m.init()
    for i in range(1, epLen+1):
        ots = []
        for k in range(m.K):
            mu_hat = m.emperical_rewards[k] / m.observations[k]
            sigma = 1/m.observations[k]
            sample = np.random.normal(mu_hat, np.sqrt(sigma))
            ots.append(max(sample,mu_hat))
        arm = np.argmax(ots)
        m.step(arm)
        r.append(m.total_reward)
    return r

def OTS_plus(m, epLen):
    r = []
    m.init()
    for i in range(1, epLen+1):
        ots_plus = []
        for k in range(m.K):
            mu_hat = m.emperical_rewards[k] / m.observations[k]
            sigma = 1/(m.observations[k] +1)
            sample = np.random.normal(mu_hat, sigma)
            bonus = np.sqrt(2 * np.log(i) / m.observations[k])
            ots_plus.append(max(sample,mu_hat + bonus))
        arm = np.argmax(ots_plus)
        m.step(arm)
        r.append(m.total_reward)
    return r

"""# Experiment"""
def main(args):
    mean_r = args.mean_r
    epLen = args.epLen
    workdir = args.workdir

    os.makedirs(workdir, exist_ok=True)

    os.chdir(workdir)
    
    with open("config.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    m = Bandits(mean_r)
    best = run_best(m, 0, epLen)
    m = Bandits(mean_r)
    ucb = UCB(m, epLen)
    ucb_regret = [a - b for a, b in zip(best, ucb)]
    m = Bandits(mean_r)
    ts = TS(m, epLen)
    ts_regret = [a - b for a, b in zip(best, ts)]
    m = Bandits(mean_r)
    ots = OTS(m, epLen)
    ots_regret = [a - b for a, b in zip(best, ots)]
    m = Bandits(mean_r)
    ots_plus = OTS_plus(m, epLen)
    ots_plus_regret = [a - b for a, b in zip(best, ots_plus)]


    linewidth = 2
    plt.plot(cum_me(ots_regret),label="OTS", linestyle="-", linewidth=2)
    plt.plot(cum_me(ots_plus_regret),label="OTS+", linestyle="dotted", linewidth=2)
    plt.plot(cum_me(ts_regret),label="TS", linestyle="--", linewidth=2)
    plt.plot(cum_me(ucb_regret), label="UCB", linestyle="-", linewidth=2)
    plt.title("K = {}".format(len(mean_r)))
    plt.xlabel("Rounds")
    plt.ylabel("Regret")
    plt.legend()
    plt.savefig("trace.png", dpi=500)
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mean_r", default=[0.1,0.05,0.03,0.02,0.01])
    # [0.1, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01]
    parser.add_argument("--epLen", default = 100000, type=int)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()
    main(args)

