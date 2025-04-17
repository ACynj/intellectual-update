'''
模拟退火算法是一种通用概率算法，常用于在一个大的搜寻空间内找寻命题的最优解。
它源于对固体退火过程的模拟，将固体加温至充分高的问题，再让其徐徐冷却，
加温时，固体内部粒子随温升变为无序状态，内能增大，而徐徐冷却时粒子渐趋有序，在每个
温度都达到平衡态，最后在常温时达到基态，内能减为最小。
在优化问题中，模拟退火算法以一定的概率接受恶化解，从而避免陷入局部最优解，
最终找到全局最优解。
'''
import numpy as np
import math


# 定义目标函数
def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

# 模拟退火算法实现
def simulated_annealing(func,dim=2,initial_temp=100,final_temp=0.1,alpha = 0.95,max_iter=100):
    # 初始化当前解
    current_solution = np.random.uniform(-5,5,dim)
    # 计算当前解的目标函数值
    current_fitness = func(current_solution)
    # 初始化最优解
    best_solution = current_solution
    best_fitness = current_fitness

    # 当前温度
    temp = initial_temp

    # 迭代过程
    while temp > final_temp:
        for _ in range(max_iter):
            # 在当前解的领域内随机生成一个新解
            new_solution = current_solution + np.random.normal(0,0.1,dim)
            # 计算新解的目标函数值
            new_fitness = func(new_solution)

            #计算目标函数值值
            delta_f = new_fitness - current_fitness

            # 判断是否接受新解
            if delta_f < 0 or np.random.rand() < math.exp(-delta_f / temp):
                # 接受新解
                current_solution = new_solution
                current_fitness = new_fitness

                # 更新最优解
                if current_fitness<best_fitness:
                    best_solution = current_solution
                    best_fitness = current_fitness

        # 温度更新
        temp*=alpha

    return best_solution,best_fitness

# 运行模拟退火算法
best_solution,best_fitness = simulated_annealing(rosenbrock)
print(f"最优解: {best_solution}")
print(f"最优适应度值: {best_fitness}")