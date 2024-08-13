import jax
import jax.numpy as jnp
from jax import vmap, lax, jit
import time

def f(i, j):
    # 任意の関数。例として i + j とする。
    return i + j

def initialize_dp(n):
    return jnp.zeros((n, n))

def compute_cumulative_sum(n):
    indices = jnp.arange(n)
    cum_sum = jnp.zeros((n, n))
    for i in range(n):
        f_vmap = vmap(lambda k: f(i, k))
        cum_sum = cum_sum.at[i, i:].set(jnp.cumsum(f_vmap(indices[i:])))
    return cum_sum

def compute_dp(n, cum_sum):
    dp = initialize_dp(n)
    
    

    
    vec_update_row = vmap(update_row)
    vec_update_all_rows = vmap(vec_update_row)
    
    for l in range(1, n):
        dp = vec_update_all_rows(dp, l)
    
    return dp

@jit
def main():
    n = 100  # 例として n = 5 とする
    cum_sum = compute_cumulative_sum(n)
    print("cum_sum calculation done")
    
    start_time = time.time()
    dp = compute_dp(n, cum_sum)
    end_time = time.time()

    t = end_time - start_time

    # 実行時間 = end_time - start_time
    print(f"計算時間: {t:.6f} 秒")
    print(dp)

if __name__ == "__main__":
    main()