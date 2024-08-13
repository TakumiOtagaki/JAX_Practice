import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt

# 通常の関数を定義
def compute(x):
    return x ** 2 + 3 * x + 2

# ベクトル化された関数を定義
vectorized_compute = jax.vmap(compute)

# JITコンパイルされたベクトル化関数
jit_vectorized_compute = jax.jit(vectorized_compute)

# さまざまな N で速度比較
N_values = [1000, 10000, 100000, 1000000]
normal_times = []
vectorized_times = []
jit_vectorized_times = []

for N in N_values:
    x = jnp.linspace(0, 100, N)
    
    # 通常の関数の実行時間を計測
    start = time.time()
    result = jnp.array([compute(xi) for xi in x])
    normal_times.append(time.time() - start)
    
    # ベクトル化された関数の実行時間を計測
    start = time.time()
    result_vectorized = vectorized_compute(x)
    vectorized_times.append(time.time() - start)
    
    # JITコンパイルされたベクトル化関数の実行時間を計測
    start = time.time()
    result_jit_vectorized = jit_vectorized_compute(x)
    jit_vectorized_times.append(time.time() - start)

# 結果をプロット
plt.plot(N_values, normal_times, label='Normal Function')
plt.plot(N_values, vectorized_times, label='Vectorized Function')
plt.plot(N_values, jit_vectorized_times, label='JIT-Vectorized Function')
plt.xlabel('N')
plt.ylabel('Time (seconds)')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.title('Performance Comparison')
plt.show()