'''
自适应大邻域搜索算法是一种启发式搜索算法，常用于解决组合优化问题。
它通过在较大的邻域内进行搜索，利用一系列破坏和修复操作来不断改进当前解。
“自适应” 体现在算法会根据操作的表现动态调整操作的使用频率，使得表现好的操作有更高的概率被选中。
'''

import numpy as np
import random



# 示例目标函数：旅行商(TSP)的距离计算
def tsp_distance(solution, distance_matrix):
    total_distance = 0
    num_cities = len(solution)
    for i in range(num_cities - 1):
        total_distance += distance_matrix[solution[i]][solution[i + 1]]
    total_distance += distance_matrix[solution[-1]][solution[0]]
    return total_distance

# 破坏操作： 随机移除k个城市
def destroy(solution, k):
    indices_to_remove = random.sample(range(len(solution)), k)
    removed_cities = [solution[i] for i in indices_to_remove]
    new_solution = [city for i,city in enumerate(solution) if i not in indices_to_remove]
    return new_solution,removed_cities

# 修复操作: 贪心插入
def repair(solution,removed_cities,distance_matrix):
    for city in removed_cities:
        best_index = 0
        best_distance = float('inf')
        # 尝试在每个位置插入被删除的城市，如果放入该位置后使得整个距离缩短，则更新
        for i in range(len(solution)+1):
            new_solution = solution.copy()
            new_solution.insert(i, city)
            distance = tsp_distance(new_solution, distance_matrix)
            if distance < best_distance:
                best_distance = distance
                best_index = i
        solution.insert(best_index,city)
    # 最后返回每次使得距离最短的解（贪心）
    return solution

# 自适应大邻域搜索算法
def alns(num_cities, distance_matrix,max_iter=500,update_interval=100,k=3,augment_alpha=1.2,augment_beta=1.1,decrease_alpha=0.9):
    # 初始化
    # 随机生成初始解
    initial_solution = list(range(num_cities))
    # 洗牌
    random.shuffle(initial_solution)
    current_solution = initial_solution
    best_solution = initial_solution
    best_distance = tsp_distance(best_solution, distance_matrix)

    # 初始化破坏和修复操作的权重
    destroy_weights = [1]
    repair_weights = [1]

    # 迭代
    for t in range(max_iter):
        # 选择操作
        destroy_index = np.random.choice(len(destroy_weights),p=np.array(destroy_weights)/sum(destroy_weights))
        repair_index = np.random.choice(len(repair_weights),p=np.array(repair_weights)/sum(repair_weights))

        # 破坏当前解
        partial_solution,removed_cities = destroy(current_solution,k)
        # 修复部分解
        new_solution = repair(partial_solution,removed_cities,distance_matrix)
        # 评估新解
        new_distance = tsp_distance(new_solution, distance_matrix)

        if new_distance < best_distance:
            # 更新最优解
            best_solution = new_solution
            best_distance = new_distance
            performance = 'improve'
        elif random.random() < 0.1: # 简单的接受准则,也可以换成别的
            current_solution = new_solution
            performance = 'acceptable'
        else:
            performance = 'rejected'


        # 更新操作权重
        if (t+1)% update_interval == 0:
            if performance == 'improve':
                destroy_weights[destroy_index]*=augment_alpha
                repair_weights[repair_index]*=augment_alpha
            elif performance == 'acceptable':
                destroy_weights[destroy_index]*=augment_beta
                repair_weights[repair_index]*=augment_beta
            else:
                destroy_weights[destroy_index]*=decrease_alpha
                repair_weights[repair_index]*=decrease_alpha

            # 归一化权重
            destroy_weights = np.array(destroy_weights) / sum(destroy_weights)
            repair_weights = np.array(repair_weights) / sum(repair_weights)
        print(f"Iter {t + 1},solution:{best_solution},Best Fitness:{best_distance:.6f}")
    return best_solution,best_distance


# 示例距离矩阵
np.random.seed(18)
num_cities = 8
distance_matrix = np.random.randint(1,100,size=(num_cities,num_cities))
np.fill_diagonal(distance_matrix,0) # 主对角线全部填充为0
print(distance_matrix)
best_solution,best_distance = alns(num_cities, distance_matrix)
print("最优解: ",best_solution)
print("最优距离: ",best_distance)