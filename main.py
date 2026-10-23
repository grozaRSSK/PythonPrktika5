import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats, spatial, interpolate
import warnings

warnings.filterwarnings('ignore')

print("Оптимизация доставки и статистический анализ с SciPy")
print("=" * 65)

# 1. СТАТИСТИЧЕСКИЙ АНАЛИЗ ДАННЫХ О ЗАКАЗАХ (модуль stats)
print("\n1. СТАТИСТИЧЕСКИЙ АНАЛИЗ ЗАКАЗОВ")
print("-" * 45)

# Генерируем данные о ежедневных заказах
np.random.seed(42)
daily_orders = np.random.normal(150, 30, 365)  # 365 дней
daily_orders = np.maximum(daily_orders, 50)  # Минимум 50 заказов

print(f"Общее количество заказов за год: {np.sum(daily_orders):.0f}")
print(f"Среднее заказов в день: {np.mean(daily_orders):.1f}")
print(f"Медиана заказов: {np.median(daily_orders):.1f}")
print(f"Стандартное отклонение: {np.std(daily_orders):.1f}")

# Проверка нормальности распределения
kurtosis = stats.kurtosis(daily_orders)
skewness = stats.skew(daily_orders)
print(f"Эксцесс: {kurtosis:.3f}")
print(f"Асимметрия: {skewness:.3f}")

# 2. ОПТИМИЗАЦИЯ ЗАПАСОВ (модуль optimize)
print("\n2. ОПТИМИЗАЦИЯ УРОВНЯ ЗАПАСОВ")
print("-" * 45)

# Функция затрат: затраты на хранение + затраты на нехватку
def inventory_cost(stock_level):
    storage_cost = 2.0   # Стоимость хранения единицы в день
    shortage_cost = 10.0 # Штраф за нехватку единицы

    # Расчет затрат для каждого дня
    daily_costs = np.where(
        daily_orders > stock_level,
        (daily_orders - stock_level) * shortage_cost,  # Затраты на нехватку
        (stock_level - daily_orders) * storage_cost    # Затраты на хранение
    )
    return np.mean(daily_costs)

# Поиск оптимального уровня запасов
result = optimize.minimize_scalar(inventory_cost, bounds=(50, 250), method='bounded')
optimal_stock = result.x
min_cost = result.fun

print(f"Оптимальный уровень запасов: {optimal_stock:.1f} единиц")
print(f"Минимальные средние суточные затраты: {min_cost:.2f} руб.")

# 3. ОПТИМИЗАЦИЯ МАРШРУТА ДОСТАВКИ (модуль spatial)
print("\n3. ОПТИМИЗАЦИЯ МАРШРУТА ДОСТАВКИ")
print("-" * 45)

# Координаты точек доставки (склад + 5 клиентов)
points = np.array([
    [0, 0],  # Склад
    [3, 4],  # Клиент 1
    [1, 7],  # Клиент 2
    [6, 2],  # Клиент 3
    [4, 5],  # Клиент 4
    [2, 3],  # Клиент 5
])

# Расчет матрицы расстояний
distance_matrix = spatial.distance_matrix(points, points)
print("Матрица расстояний между точками:")
print(np.round(distance_matrix, 2))

# Функция для расчета общей длины маршрута
def route_length(route):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i], route[i + 1]]
    # Возвращаемся на склад
    total_distance += distance_matrix[route[-1], route[0]]
    return total_distance

# Простой перебор для небольшого количества точек
from itertools import permutations
all_routes = list(permutations(range(1, len(points))))  # Все возможные маршруты
best_route = None
best_length = float('inf')

for route in all_routes:
    full_route = (0,) + route + (0,)  # Начинаем и заканчиваем на складе
    length = route_length(full_route)
    if length < best_length:
        best_length = length
        best_route = full_route

print(f"Оптимальный маршрут: {best_route}")
print(f"Длина оптимального маршрута: {best_length:.2f} км")

# 4. ИНТЕРПОЛЯЦИЯ ГРАФИКА ЗАКАЗОВ (модуль interpolate)
print("\n4. ПРОГНОЗИРОВАНИЕ ЗАКАЗОВ")
print("-" * 45)

# Данные за первые 30 дней
days = np.arange(30)
orders_first_month = daily_orders[:30]

# Кубическая интерполяция для прогнозирования
interp_func = interpolate.interp1d(days, orders_first_month, kind='cubic', fill_value='extrapolate')
future_days = np.arange(40)  # Прогноз на 40 дней
predicted_orders = interp_func(future_days)

print("Прогноз заказов на основе первых 30 дней:")

# 5. АНАЛИЗ НАГРУЗКИ НА КУРЬЕРОВ
print("\n5. АНАЛИЗ НАГРУЗКИ НА КУРЬЕРОВ")
print("-" * 45)

# Распределение количества заказов на курьера
orders_per_courier = daily_orders / 10  # 10 курьеров
mean_orders = np.mean(orders_per_courier)
std_orders = np.std(orders_per_courier)

# Вероятность перегрузки (более 20 заказов на курьера)
overload_prob = np.mean(orders_per_courier > 20)
print(f"Средняя нагрузка на курьера: {mean_orders:.1f} заказов")
print(f"Стандартное отклонение нагрузки: {std_orders:.1f}")
print(f"Вероятность перегрузки (>20 заказов): {overload_prob:.3f}")

# 6. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
print("\n6. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("-" * 45)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Распределение заказов
axes[0, 0].hist(daily_orders, bins=30, density=True, alpha=0.7, color='lightblue', edgecolor='black')
axes[0, 0].axvline(optimal_stock, color='red', linestyle='--', linewidth=2,
                   label=f'Оптимальный запас: {optimal_stock:.1f}')
axes[0, 0].set_xlabel('Количество заказов в день')
axes[0, 0].set_ylabel('Плотность вероятности')
axes[0, 0].set_title('Распределение ежедневных заказов')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Функция затрат
stock_levels = np.linspace(50, 250, 100)
costs = [inventory_cost(level) for level in stock_levels]
axes[0, 1].plot(stock_levels, costs, 'b-', linewidth=2)
axes[0, 1].plot(optimal_stock, min_cost, 'ro', markersize=8, label=f'Оптимум: {optimal_stock:.1f}')
axes[0, 1].set_xlabel('Уровень запасов')
axes[0, 1].set_ylabel('Средние суточные затраты (руб.)')
axes[0, 1].set_title('Оптимизация уровня запасов')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Карта маршрута
axes[0, 2].scatter(points[:, 0], points[:, 1], s=100, c='red', marker='o')
for i, (x, y) in enumerate(points):
    axes[0, 2].text(x + 0.1, y + 0.1, f'{i}', fontweight='bold')
# Рисуем оптимальный маршрут
route_points = points[list(best_route)]
axes[0, 2].plot(route_points[:, 0], route_points[:, 1], 'b-', linewidth=2, marker='o')
axes[0, 2].set_xlabel('X координата')
axes[0, 2].set_ylabel('Y координата')
axes[0, 2].set_title('Оптимальный маршрут доставки')
axes[0, 2].grid(True, alpha=0.3)

# 4. Прогнозирование заказов
axes[1, 0].plot(days, orders_first_month, 'bo-', label='Фактические данные', markersize=4)
axes[1, 0].plot(future_days, predicted_orders, 'r--', label='Прогноз')
axes[1, 0].set_xlabel('День')
axes[1, 0].set_ylabel('Количество заказов')
axes[1, 0].set_title('Прогнозирование заказов')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Нагрузка на курьеров
axes[1, 1].hist(orders_per_courier, bins=20, density=True, alpha=0.7, color='orange', edgecolor='black')
axes[1, 1].axvline(20, color='red', linestyle='--', linewidth=2, label='Порог перегрузки (20)')
axes[1, 1].set_xlabel('Заказов на курьера')
axes[1, 1].set_ylabel('Плотность вероятности')
axes[1, 1].set_title('Распределение нагрузки на курьеров')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Ежедневная динамика (первые 60 дней)
axes[1, 2].plot(np.arange(60), daily_orders[:60], 'g-', linewidth=1.5)
axes[1, 2].set_xlabel('День')
axes[1, 2].set_ylabel('Количество заказов')
axes[1, 2].set_title('Ежедневная динамика заказов (первые 60 дней)')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()