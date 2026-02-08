import numpy as np, pathlib
p = pathlib.Path(__file__).resolve().parent.parent / "data" / "mini_ndvi_stack.npz"
d = np.load(p)
for k in d.files:
    a = d[k]
    print(k, a.shape, a.dtype)
    if a.ndim == 1:
        print("  ", a[:3])
