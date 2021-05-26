import numpy as np

def kernel(x, y):
    # k = np.cos(x)**2 * np.cos(y)**2 + np.sin(x)**2 * np.sin(y)**2 + np.cos(x) * np.cos(y) * np.sin(x) * np.sin(y) *(2 * np.cos(x-y))
    k = np.cos(x-y)**2
    return k

def center(K):
    n = len(K)
    K = 1/ n* np.array(K)
    one_n = 1 / n * np.ones((n,n))
    K_cent = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    return K_cent

X = np.random.uniform(0,1, 100)

K = np.array([[kernel(x,y) for y in X]for x in X])

print(np.linalg.eigvalsh(K)[-4:])