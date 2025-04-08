import numpy as np

# 示例：优化Sphere函数
def sphere_function(x):
    return sum(x**2)

def differential_evolution(func,bounds,NP=50,F=0.8,CR=0.9,max_iter=100):
    '''
    差分进化算法实现

    参数：
    func：目标函数 最小化
    bounds: 每个维度的上下界列表 例如：[(min1,max1),(min2,max2),...]
    NP: 种群大小 默认50
    F: 缩放因子 默认0.8
    CR：交叉概率 默认0.9
    max_iter:最大迭代次数 默认100

    返回：
    best_solution:最优解
    best_fitness: 最优适应度
    '''

    # 1、初始化参数
    D = len(bounds) # 问题维度
    min_bounds = np.array([b[0] for b in bounds])
    max_bounds = np.array([b[1] for b in bounds])

    # 初始化种群
    population = min_bounds + np.random.rand(NP,D) * (max_bounds - min_bounds)
    fitness = np.array([func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx]
    best_fitness = fitness[best_idx]

    for iter in range(max_iter):
        for i in range(NP):
            # 2、变异：随机选择三个不同的个体
            idxs = [idx for idx in range(NP) if idx!=i]
            r1,r2,r3 = np.random.choice(idxs,3,replace = False)
            mutant = population[r1] + F*(population[r2] - population[r3])

            # 确保突变体在边界内
            mutant = np.clip(mutant,min_bounds,max_bounds)
            # 3、交叉:生成试验个体
            cross_points = np.random.rand(D) < CR
            cross_points[np.random.randint(D)] = True # 至少一个维度交叉,避免所有维度不交叉的极端情况
            trial = np.where(cross_points,mutant,population[i]) # 根据维度的bool值，选择变异体和原个体该维度的值

            # 选择，贪婪保留更优解
            trial_fitness = func(trial)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial.copy()
        print(f"Iter {iter + 1},solution:{best_solution},Best Fitness:{best_fitness:.6f}")
    return best_solution,best_fitness

if __name__ == "__main__":
    bounds = [(-5,5)] * 10
    best_sol,best_val = differential_evolution(sphere_function,bounds,max_iter=100)
    print(f"\n最优解:{best_sol}\n最优值{best_val}")
