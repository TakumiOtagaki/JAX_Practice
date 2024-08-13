import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt

# 2つの数値の和を計算する関数を定義
def add(x, y):
    return x + y

# JITコンパイルされた関数
add_jit = jax.jit(add)

# さまざまな N の値
N_values = [int(1e3), int(1e4), int(1e5), int(1e6), int(1e7)]
times_without_jit = []
times_with_jit_first = []
times_with_jit_subsequent = []

for N in N_values:
    x = jnp.ones((N,))
    y = jnp.ones((N,))

    # 通常の関数の実行時間を計測
    start = time.time()
    result = add(x, y)
    times_without_jit.append(time.time() - start)

    # JITコンパイルされた関数の初回実行時間を計測
    start = time.time()
    result_jit = add_jit(x, y)
    times_with_jit_first.append(time.time() - start)

    # JITコンパイルされた関数の再実行時間を計測
    start = time.time()
    result_jit = add_jit(x, y)
    times_with_jit_subsequent.append(time.time() - start)

# グラフを作成
plt.figure(figsize=(10, 6))
plt.plot(N_values, times_without_jit, label='Without JIT')
plt.plot(N_values, times_with_jit_first, label='With JIT (first run)')
plt.plot(N_values, times_with_jit_subsequent, label='With JIT (subsequent runs)')
plt.xlabel('array size N')
plt.ylabel('Time (seconds)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title('Performance Comparison with and without JIT')

# 右の軸に比率をプロット
ax1 = plt.gca()
ax2 = ax1.twinx()
ratio = [t2 / t1 for t1, t2 in zip(times_with_jit_subsequent, times_without_jit)]
ax2.plot(N_values, ratio, 'r*', label='With JIT / Without JIT')
ax2.set_ylabel('Ratio (Without JIT / With JIT)')
ax2.set_yscale('log')
# ax2.legend(loc='upper left')

plt.savefig("/large/otgk/JAX_Practice/graphs/tutorials_time.png")