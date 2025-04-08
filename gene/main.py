# 目标函数 x为0到31的整数(5为二进制编码)
import string


def fitness_function(x):
    return x**2

import random
import numpy as np
# ======= 参数设置 =======
POPULATION_SIZE = 50 # 种群大小
CHROMOSOME_LENGTH = 5 # 染色体长度(二进制编码位数)
CROSSOVER_RATE = 0.8 # 交叉概率
MUTATION_RATE = 0.01 # 变异概率
MAX_GENERATIONS = 100 # 最大迭代次数

#======= 辅助函数 =======
def decimal_to_binary(x): # 十进制转二进制
    return format(x,'0{}b'.format(CHROMOSOME_LENGTH))

def binary_to_decimal(binary_str): # 二进制转十进制
    return int(binary_str,2)

# ======= 初始化种群 =======
def initialize_population():
    '''生成随机二进制编码的初始种群'''
    population = []
    for _ in range(POPULATION_SIZE):
        # 生成随机5位二进制数（0-31）
        individual = ''.join(random.choice('01') for _ in range(CHROMOSOME_LENGTH))
        population.append(individual)
    return population
# ======= 适应度计算 =======
def calculate_fitness(population):
    # 计算种群中每个个体的适应度
    fitness = []
    for individual in population:
        # 二进制转为十进制
        x = binary_to_decimal(individual)
        fitness.append(fitness_function(x))
    return fitness

# ======= 选择操作(轮盘赌) =======
'''适应度越高的个体占据轮盘上越大的区域，被随机数命中的概率越大。'''
def select_parents(population,fitness):
    ''' 根据适应度选择两个父代 '''
    total_fitness = sum(fitness)
    # 计算累积概率分布
    probabilities = [f/total_fitness for f in fitness]
    # 累积概率
    cumulative_prob = np.cumsum(probabilities)

    # 选择两个父代
    parents = []
    for _ in range(2):
        rand = random.random()
        for i,cp in enumerate(cumulative_prob):
            if rand < cp:
                parents.append(population[i])
                break
    return parents

# ======= 交叉操作(单点交叉:将交叉点之后的基因片段交换) =======
def crossover(parent1,parent2):
    ''' 单点交叉生成两个子代 '''
    if random.random() < CROSSOVER_RATE:
        # 随机选择交叉点(1~CHROMOSOME_LENGTH-1)
        crossover_point = random.randint(1,CHROMOSOME_LENGTH-1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    else:
        return parent1, parent2

# ======= 变异操作(位翻转) =======
def mutate(child):
    '''每个基因位按位概率翻转'''
    mutated = list(child)
    for i in range(len(mutated)):
        if random.random() < MUTATION_RATE:
            mutated[i] = '1' if mutated[i] == '0' else '0'
    return ''.join(mutated)
# ======= 主算法流程 =======
def genetic_algorithm():
    # 初始化种群
    population = initialize_population()
    best_fitness_history = [] # 记录每代最佳适应度
    for generation in range(MAX_GENERATIONS):
        # 计算适应度
        fitness = calculate_fitness(population)
        best_fitness = max(fitness)
        best_fitness_history.append(best_fitness)

        # 生成新一代种群
        new_population = []

        # 保留精英（最优个体直接进入下一代）
        best_index = fitness.index(best_fitness)
        new_population.append(population[best_index])

        # 生成剩余个体
        while(len(new_population) < POPULATION_SIZE):
            # 选择父代
            parents = select_parents(population,fitness)
            # 交叉生成子代
            child1,child2 = crossover(parents[0],parents[1])
            # 变异
            child1 = mutate(child1)
            child2 = mutate(child2)

            # 加入新种群
            new_population.extend([child1,child2])

        # 打印进度
        if generation % 10 == 0:
            best_x = binary_to_decimal(population[best_index])
            print(f"Generation {generation}: Best x = {best_x},f(x)={best_fitness}")

        # 确保种群大小不变
        population = new_population[:POPULATION_SIZE]


    # 最终结果
    best_individual = population[fitness.index(max(fitness))]
    best_x = binary_to_decimal(best_individual)
    return best_x,best_fitness_history

if __name__ == "__main__":
    best_x,history = genetic_algorithm()
    print("\n=========最终结果=============")
    print(f"最优解 x = {best_x},f(x) = {fitness_function(best_x)}")