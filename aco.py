'''
蚁群算法模拟蚂蚁觅食行为，通过信息素的沉积和挥发机制寻找最优路径。
蚂蚁在移动过程中释放信息素，后续蚂蚁倾向于选择信息素浓度高的路径，形成正反馈机制。
同时，信息素会随时间挥发，避免算法陷入局部最优。
'''
import numpy as np
def calculate_path_length(path,distance_matrix):
    """计算路径总长度"""
    length = 0
    for i in range(len(path)-1):
        length+=distance_matrix[path[i]][path[i+1]]
    return length

def ant_colony_optimization(distance_matrix,num_ants,iterations,alpha,beta,rho,Q):
    """
    参数：
    distance_matrix: 城市间距离矩阵
    num_ants: 蚂蚁数量
    iterations: 迭代次数
    alpha: 信息素权重
    beta: 启发式信息权重
    rho: 信息素挥发率
    Q： 信息素强度常数

    返回：
        best_path: 最优路径
        best_length: 最短距离
    """
    n_cities = distance_matrix.shape[0]
    # 初始化信息素矩阵
    pheromone = np.ones((n_cities,n_cities))
    best_path = None
    best_length = float('inf')

    for iteration in range(iterations):
        all_paths = []
        all_lengths = []

        # 每只蚂蚁构建路径
        for ant in range(num_ants):
            path = [0] # 起点位城市 0
            visited = set({0})
            current_city = 0

            while len(path) < n_cities:
                allowed = [j for j in range(n_cities) if j not in visited]
                # 都访问过了，则退出循环
                if not allowed:
                    break
                # 计算转移概率
                probabilities = []
                for j in allowed:
                    ph = pheromone[current_city,j] # 信息素浓度
                    heu = 1.0/distance_matrix[current_city,j] # 启发式信息
                    probabilities.append((ph ** alpha) * (heu **beta))

                # 处理0概率问题
                total = sum(probabilities)
                if total == 0: # 分配均匀分布，所有概率为等概率
                    probabilities = [1/len(allowed)]*len(allowed)
                else:
                    probabilities = [p/total for p in probabilities]

                next_city = np.random.choice(allowed, p=probabilities)
                path.append(next_city)
                visited.add(next_city)
                current_city = next_city

            # 总路径长度
            path.append(0)
            length = calculate_path_length(path,distance_matrix)
            all_paths.append(path)
            all_lengths.append(length)

            # 更新全局最优
            if length < best_length:
                best_length = length
                best_path = path

        # 信息素挥发
        pheromone *= (1-rho)

        # 更新信息素增量
        delta_pheromone = np.zeros_like(pheromone)
        # 对所有蚂蚁走的路径进行遍历
        for path,length in zip(all_paths,all_lengths):
            for i in range(len(path)-1):
                city_a,city_b = path[i],path[i+1]
                # 路径越短‌，蚂蚁往返越快，‌单位时间内留下的信息素浓度越高‌
                delta_pheromone[city_a,city_b] += Q/length
                delta_pheromone[city_b, city_a] += Q / length

        # 更新信息素
        pheromone += delta_pheromone
        print(f"Iteration {iteration + 1}: best:{best_length},path:{best_path}")

    return best_path, best_length

if __name__ == "__main__":
    distance_matrix = np.array([
        [0,10,15,20],
        [10,0,35,25],
        [15,35,0,30],
        [20,25,30,0]
    ])
    best_path,best_length =  ant_colony_optimization(distance_matrix,10,100,1,2,0.1,100)
    print("最优路径: ",best_path)
    print("最短距离: ",best_length)