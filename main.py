import matplotlib.pyplot as plt
import numpy as np

# np.random.seed(10)
# 假设赌博机的奖励服从正态分布
E = np.array([1, 9, 3, 2, 7])  # 赌博机均值
D = np.array([5, 3, 1, 7, 4])  # 赌博机方差
N = 5  # 赌博机数量
T = 1000  # 游戏轮数

ESTIMATED_Q = np.ones(N)  # 奖励的估计值

CHOOSE_TIMES = np.ones(N)  # UCB每个老虎机被选择的次数
UCB_SCORE = np.ones(N)  # UCB每个老虎机的得分At

A = np.ones(N)  # thompson sampling beta分布参数a
B = np.ones(N)  # thompson sampling beta分布参数b

H = np.ones(N)  # gradient bandit 的偏好函数


def epsilon_greedy(step):
    rewards = np.random.normal(E, D)
    rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))  # 归一化
    if np.random.rand() < 0.9:  # 90%贪婪选择
        choose_action = np.argmax(ESTIMATED_Q)
    else:
        choose_action = np.random.randint(0, N)
    # 更新Qn+1 平稳情况
    ESTIMATED_Q[choose_action] = ESTIMATED_Q[choose_action] + (
            rewards[choose_action] - ESTIMATED_Q[choose_action]) / step
    return rewards[choose_action]


def ucb(step):
    rewards = np.random.normal(E, D)
    rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))  # 归一化
    choose_action = np.argmax(UCB_SCORE)  # 根据At选择行动
    ESTIMATED_Q[choose_action] = ESTIMATED_Q[choose_action] + (
            rewards[choose_action] - ESTIMATED_Q[choose_action]) / step
    CHOOSE_TIMES[choose_action] += 1
    UCB_SCORE[choose_action] = ESTIMATED_Q[choose_action] + np.sqrt(8 * np.log(step) / CHOOSE_TIMES[choose_action])
    return rewards[choose_action]


def thompson_sampling():
    rewards = np.random.normal(E, D)
    rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))  # 归一化
    beta_value = np.random.beta(A, B)
    choose_action = np.argmax(beta_value)  # 根据beta值选择行动
    A[choose_action] += rewards[choose_action]
    B[choose_action] += 1 - rewards[choose_action]
    return rewards[choose_action]


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def gradient_bandit(step):
    rewards = np.random.normal(E, D)
    rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))  # 归一化
    pi = softmax(H)
    choose_action = np.argmax(pi)
    ESTIMATED_Q[choose_action] = ESTIMATED_Q[choose_action] + (
            rewards[choose_action] - ESTIMATED_Q[choose_action]) / step
    for i in range(N):
        if i == choose_action:
            H[i] += (rewards[choose_action] - 0.8 * ESTIMATED_Q[choose_action]) * (1 - pi[choose_action])
        else:
            H[i] -= (rewards[choose_action] - 0.8 * ESTIMATED_Q[choose_action]) * pi[choose_action]
    return rewards[choose_action]


def play_game(method):
    global ESTIMATED_Q
    ESTIMATED_Q = np.ones(N)
    score = 0
    avg_score_list = []
    for i in range(T):
        if method == 'epsilon_greedy':
            r = epsilon_greedy(i + 1)
        elif method == 'ucb':
            r = ucb(i + 1)
        elif method == 'thompson_sampling':
            r = thompson_sampling()
        else:
            r = gradient_bandit(i + 1)
        score += r
        avg_score_list.append(score / (i + 1))

    return score, avg_score_list


if __name__ == '__main__':
    _, e_greedy = play_game('epsilon_greedy')
    _, ucb = play_game('ucb')
    _, thompson_sampling = play_game('thompson_sampling')
    _, gradient_bandit = play_game('gradient_bandit')

    fig = plt.figure(figsize=(8, 6))

    plt.plot(e_greedy, color='green', label='epsilon greedy')
    plt.plot(thompson_sampling, color='red', label='thompson sampling')
    plt.plot(ucb, color='blue', label='ucb')
    plt.plot(gradient_bandit, color='black', label='gradient bandit')

    plt.legend()
    plt.xlabel('iteration times')
    plt.ylabel('average scores')
    plt.show()
