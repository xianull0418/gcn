import numpy as np
import torch
from ..config import Config

class PSOGWO:
    def __init__(self, config: Config, fitness_func):
        self.config = config
        self.fitness_func = fitness_func
        self.population_size = config.POPULATION_SIZE
        self.max_iter = config.MAX_ITERATIONS
        self.lambda_ = config.LAMBDA
        self.alpha = config.ALPHA
        self.beta = config.BETA
        
    def initialize_population(self, dim, bounds):
        """初始化种群"""
        population = np.random.uniform(
            bounds[:, 0], 
            bounds[:, 1], 
            (self.population_size, dim)
        )
        return population
    
    def update_wolf(self, pos, alpha, beta, delta, a):
        """更新狼的位置"""
        r1 = np.random.random()
        r2 = np.random.random()
        A = 2 * a * r1 - a
        C = 2 * r2
        
        D_alpha = abs(C * alpha - pos)
        D_beta = abs(C * beta - pos)
        D_delta = abs(C * delta - pos)
        
        X1 = alpha - A * D_alpha
        X2 = beta - A * D_beta
        X3 = delta - A * D_delta
        
        return (X1 + X2 + X3) / 3
    
    def optimize(self, dim, bounds):
        """执行PSO-GWO混合优化"""
        print(f"开始优化 {dim} 个参数")
        print(f"参数范围: \n{bounds}")
        
        # 初始化种群
        print("\n初始化种群...")
        population = self.initialize_population(dim, bounds)
        print(f"种群大小: {len(population)}")
        
        print("\n评估初始种群适应度...")
        fitness = np.array([self.fitness_func(p) for p in population])
        
        # 初始化alpha, beta, delta狼
        sorted_idx = np.argsort(fitness)
        alpha_pos = population[sorted_idx[0]].copy()
        beta_pos = population[sorted_idx[1]].copy()
        delta_pos = population[sorted_idx[2]].copy()
        
        print(f"\n初始最优解: {alpha_pos}")
        print(f"初始最优适应度: {fitness[sorted_idx[0]]}")
        
        # 初始化速度
        velocity = np.zeros((self.population_size, dim))
        
        # 优化迭代
        for iter in range(self.max_iter):
            print(f"\n迭代 {iter+1}/{self.max_iter}")
            a = 2 - iter * (2 / self.max_iter)  # 线性递减
            
            for i in range(self.population_size):
                # GWO部分
                wolf_pos = self.update_wolf(
                    population[i], 
                    alpha_pos, 
                    beta_pos, 
                    delta_pos, 
                    a
                )
                
                # PSO部分
                r1, r2 = np.random.random(2)
                cognitive = self.alpha * r1 * (alpha_pos - population[i])
                social = self.beta * r2 * (beta_pos - population[i])
                velocity[i] = self.lambda_ * velocity[i] + cognitive + social
                
                # 混合更新
                new_pos = population[i] + velocity[i]
                
                # 边界处理
                new_pos = np.clip(new_pos, bounds[:, 0], bounds[:, 1])
                
                # 更新位置
                new_fitness = self.fitness_func(new_pos)
                if new_fitness < fitness[i]:
                    population[i] = new_pos
                    fitness[i] = new_fitness
            
            # 更新alpha, beta, delta狼
            sorted_idx = np.argsort(fitness)
            alpha_pos = population[sorted_idx[0]].copy()
            beta_pos = population[sorted_idx[1]].copy()
            delta_pos = population[sorted_idx[2]].copy()
            
            print(f"当前最优适应度: {fitness[sorted_idx[0]]:.6f}")
            print(f"当前最优解: {alpha_pos}")
        
        return alpha_pos, fitness[sorted_idx[0]] 