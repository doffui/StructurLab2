import numpy as np
import time
import random

N_SMALL = 200
N_LARGE = 4096
BLOCK_SIZE = 512

np.random.seed(42)

A_small = np.array([[random.random() for _ in range(N_SMALL)] for _ in range(N_SMALL)], dtype=np.float64)
B_small = np.array([[random.random() for _ in range(N_SMALL)] for _ in range(N_SMALL)], dtype=np.float64)

A_large = np.random.rand(N_LARGE, N_LARGE).astype(np.float64)
B_large = np.random.rand(N_LARGE, N_LARGE).astype(np.float64)

complexity_small = 2 * N_SMALL ** 3
complexity_large = 2 * N_LARGE ** 3

def measure_performance(func, *args):
    start_time = time.time()
    result = func(*args)
    elapsed_time = time.time() - start_time
    mflops = complexity_small / (elapsed_time * 1e6) if args[0].shape[0] == N_SMALL else complexity_large / (elapsed_time * 1e6)
    return result, elapsed_time, mflops

print("Работу выполнила: Захарова Анастасия Григорьевна 09.03.01ПОВа-o24")

print("1-Й ВАРИАНТ: УМНОЖЕНИЕ ПО ФОРМУЛЕ ИЗ ЛИНЕЙНОЙ АЛГЕБРЫ")
print(f"Размер матрицы: {N_SMALL}x{N_SMALL}")

start_time_1 = time.time()
result1 = np.zeros((N_SMALL, N_SMALL), dtype=np.float64)

for i in range(N_SMALL):
    for k in range(N_SMALL):
        aik = A_small[i, k]
        for j in range(N_SMALL):
            result1[i, j] += aik * B_small[k, j]

time1 = time.time() - start_time_1
mflops1 = complexity_small / (time1 * 1e6)

print(f"Время выполнения: {time1:.2f} секунд")
print(f"Производительность: {mflops1:.2f} MFLOPS")

print("2-Й ВАРИАНТ: ИСПОЛЬЗОВАНИЕ BLAS (numpy.dot)")
print(f"Размер матрицы: {N_LARGE}x{N_LARGE}")

C_blas, time_blas, mflops_blas = measure_performance(lambda x, y: np.dot(x, y), A_large, B_large)

print(f"Время выполнения: {time_blas:.2f} секунд")
print(f"Производительность: {mflops_blas:.2f} MFLOPS")

# --- 3. Блочное умножение (NumPy, блоки) ---
print("3-Й ВАРИАНТ: ОПТИМИЗИРОВАННЫЙ АЛГОРИТМ (блоковое умножение)")
print(f"Размер матрицы: {N_LARGE}x{N_LARGE}")

def matrix_multiply_optimized(A, B, block_size=BLOCK_SIZE):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=A.dtype)
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            for k in range(0, n, block_size):
                C[i:i+block_size, j:j+block_size] += np.dot(
                    A[i:i+block_size, k:k+block_size],
                    B[k:k+block_size, j:j+block_size]
                )
    return C

start_time_3 = time.time()
C_optimized = matrix_multiply_optimized(A_large, B_large)
time3 = time.time() - start_time_3
mflops3 = complexity_large / (time3 * 1e6)
performance_ratio = mflops3 / mflops_blas

print(f"Время выполнения: {time3:.2f} секунд")
print(f"Производительность: {mflops3:.2f} MFLOPS")


print(f"1-й вариант (размер {N_SMALL}x{N_SMALL}): {mflops1:.2f} MFLOPS")
print(f"2-й вариант (размер {N_LARGE}x{N_LARGE}): {mflops_blas:.2f} MFLOPS")
print(f"3-й вариант (размер {N_LARGE}x{N_LARGE}): {mflops3:.2f} MFLOPS")
print(f"Отношение производительности (3-й / 2-й): {performance_ratio:.2f}")
