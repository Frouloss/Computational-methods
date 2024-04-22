import numpy as np
import matplotlib.pyplot as plt

# Определение функции
def f(x1, x2):
    return -3*x1**2 - 42*x1 - 2*x2**2 - 12*x2 - 158

# Вычисление градиента функции
def gradient(x1, x2):
    df_dx1 = -6*x1 - 42
    df_dx2 = -4*x2 - 12
    return np.array([df_dx1, df_dx2])

import numpy as np

# Функция для метода случайного поиска
def random_search(x_start, num_iterations, tolerance = 0.0001):
    best_solution = None
    best_score = 0
    f_max = 7
    trajectory = [x_start]
    #while abs(f_max - best_score) < tolerance:
    for _ in range(num_iterations):
        solution = np.array([np.random.uniform(-8, 8), np.random.uniform(-8, 8)])
        score = f(*solution)
        if score > best_score:
            best_solution = solution
            best_score = score
            trajectory.append(best_solution)
    return best_solution, best_score, np.array(trajectory)

# Количество итераций для метода случайного поиска
num_iterations = 10000
x_start = np.array([1, 8])  # Начальное приближение

# Запуск метода случайного поиска
random_solution, random_score, random_trajectory = random_search(x_start, num_iterations)

print(random_trajectory)

# Вывод результатов
print("/n Метод случайного поиска: ")
print("Точка максимума: ", random_solution)
print("Значение функции в максимуме: ", random_score)


# Градиентный метод с дроблением шага
def gradient_descent(x, alpha, epsilon, iter):
    trajectory = [x]
    while True:
        grad = gradient(x[0], x[1])
        # Выбор начального значения шага
        step_size = alpha
        # Дробление шага
        while f(x[0] + step_size * grad[0], x[1] + step_size * grad[1]) < f(x[0], x[1]):
            step_size *= 0.5
        # Обновление значений переменных
        x_new = x + step_size * grad
        # Проверка условия остановки
        if np.linalg.norm(x_new - x) < epsilon:
            break
        x = x_new
        trajectory.append(x)
        iter += 1
    return np.array(trajectory), iter

# Метод Хука-Дживса
def hooke_jeeves(initial_guess, step_sizes=None, tolerance=1e-6, iter = 0):
    n = len(initial_guess)
    if step_sizes is None:
        step_sizes = [1.0] * n
    x = np.array(initial_guess)
    trajectory = [x]
    while True:
        x_best = x.copy()
        f_best = f(x[0], x[1])
        improved = False
        for i in range(n):
            for delta in [-step_sizes[i], 0, step_sizes[i]]:
                x_test = x.copy()
                x_test[i] += delta
                f_test = f(x_test[0], x_test[1])
                if f_test > f_best:
                    x_best = x_test
                    f_best = f_test
                    improved = True
        if not improved:
            break
        x = x_best
        trajectory.append(x)
        iter += 1
    return np.array(trajectory), iter

# Исходные значения переменных и параметры
x_start = np.array([1, 8])  # Начальное приближение
alpha = 0.01  # Начальное значение шага
epsilon = 0.0001  # Порог сходимости

# Запуск градиентного метода
iter = 0
trajectory, iter = gradient_descent(x_start, alpha, epsilon, iter)
print("\nМетод Градиентного спуска с дроблением шага:")
print("Точка максимума:", trajectory[-1])
print("Значение функции в максимуме: ", f(trajectory[-1][0], trajectory[-1][1]))
print("Число итераций:", iter)

# Запуск метода Хука-Дживса
iter = 0
hooke_score, iter = hooke_jeeves(x_start)
print("\nМетод Хука-Дживса:")
print("Точка максимума: ", hooke_score[-1])
print("Значение функции в максимуме: ", f(hooke_score[-1][0], hooke_score[-1][1]))
print("Число итераций:", iter)

# Создание сетки точек для построения линий уровня
x1_values = np.linspace(-10, 10, 400)
x2_values = np.linspace(-10, 10, 400)
X1, X2 = np.meshgrid(x1_values, x2_values)
Z = f(X1, X2)

# Построение линий уровня и траектории поиска
plt.figure(figsize=(8, 6))
plt.contour(X1, X2, Z, levels=20)
plt.plot(trajectory[:, 0], trajectory[:, 1], color='r', label='Метод градиента с дроблением шага')
plt.plot(hooke_score[:, 0], hooke_score[:, 1], color='g', label='Метод Хука-Дживса')
plt.plot(random_trajectory[:, 0], random_trajectory[:, 1], color='y', label='Метод случайного поиска')
plt.scatter(x_start[0], x_start[1], marker='o', color='r', label='Начальная точка')
plt.title('Линии уровня и траектория поиска')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.colorbar()
plt.show()


