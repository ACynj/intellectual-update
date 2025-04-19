'''
粒子群算法（Particle Swarm Optimization, PSO）是一种基于群体智能的优化算法，它模拟了鸟群或鱼群的群体行为。
在 PSO 中，每个个体被称为一个 “粒子”，每个粒子代表问题的一个潜在解。粒子在搜索空间中飞行，
通过跟踪个体的历史最优位置（pbest）和整个群体的历史最优位置（gbest）来更新自己的位置。
'''
import numpy as np

# 作业要求的函数
def f(x):
    x1, x2 = x
    sum1 = 0
    sum2 = 0
    for i in range(1, 6):
        sum1 += i * np.cos((i + 1) * x1 + i)
        sum2 += i * np.cos((i + 1) * x2 + i)
    return sum1 * sum2

# 定义目标函数(这里以Rosenbrock函数为例)
def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0)**2.0+(1-x[:-1])**2.0)

# 粒子群算法实现
def pso(func,# 目标函数，即需要优化的函数
        dim=2, # 解的维度
        num_particles=80, # 粒子的数量，粒子越多搜索范围越广，但计算量也会增大
        max_iter=500, # 最大迭代次数
        w=0.6,# 惯性权重，控制粒子先前速度对当前速度的影响程度
        c1=4.3, # 个体学习因子，调节粒子向自身历史最优位置飞行的步长
        c2=10.5): # 社会学习因子，调节粒子向全局历史最优位置飞行的步长

    # 加入种子，保证代码可重复性
    np.random.seed(42)

    # 初始化粒子的位置和速度
    particles_position = np.random.uniform(-10,10,(num_particles,dim)) # 生成-5<=position<5服从均匀分布的随机数，生成的维度为num_particles*dim
    particles_velocity = np.random.uniform(-1,1,(num_particles,dim))

    #初始化个体最优位置和全局最优位置
    # 个体最优
    particles_pbest = particles_position.copy()
    particles_pbest_fitness = np.array([func(p) for p in particles_position])
    # 全局最优
    gbest_index = np.argmax(particles_pbest_fitness)
    gbest = particles_pbest[gbest_index]
    gbest_fitness = particles_pbest_fitness[gbest_index]

    # 迭代更新
    for iter in range(max_iter):
        # 对每个粒子更新速度和位置
        for i in range (num_particles):
            # 更新速度
            r1,r2 = np.random.rand(2)
            particles_velocity[i] = (w*particles_velocity[i] + c1*r1*(particles_pbest[i] - particles_position[i])+ c2 *r2*(gbest - particles_position[i]))

            # 更新位置
            particles_position[i] += particles_velocity[i]

            # 计算新的适应度值
            fitness = func(particles_position[i])

            # 更新个体最优位置
            if fitness > particles_pbest_fitness[i]:
                particles_pbest[i] = particles_position[i]
                particles_pbest_fitness[i] = fitness

                # 更新全局位置
                if fitness > gbest_fitness:
                    gbest = particles_position[i]
                    gbest_fitness = fitness
        print(f"Iter {iter + 1},solution:{gbest},Best Fitness:{gbest_fitness:.6f}")
    return gbest,gbest_fitness


# 运行粒子群算法
best_solution,best_fitness = pso(f)
print(f"最优解: {best_solution}")
print(f"最优适应度值: {best_fitness}")
