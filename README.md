# image_completion
This is my final project submission for CS 539, Convex Optimization.
![Header Image](figs/fig_cmp.png?raw=true "Image Recovery")

# Setup
I ran these scripts on Ubuntu 18.04. You will need these packages,
```bash
pip3 install cvxpy python-mnist cifar10_web --user
```

If you get an error that includes something like this,
```bash
FATAL: Cannot solve SDPs with > 2x2 matrices without linked blas+lapack libraries
Install blas+lapack and re-compile SCS with blas+lapack library locations
ERROR: init_cone failure
Failure:could not initialize work
```

Then this might fix your issue:
```bash
sudo apt install -y libatlas-base-dev
pip3 install --no-cache-dir --ignore-installed scs --user
```


# Usage

To display some example images from both datasets,
```python
python3 show_examples.py
```

To reproduce the results shown in Fig. 2 of [Candes et. al.](https://arxiv.org/abs/0805.4471),
```python
python3 candes_results.py
```

To perform matrix recovery on a subset of MNIST,
```python
python3 run_mnist.py
```

To perform matrix recovery on a subset of CIFAR-10,
```python
python3 run_cifar.py
```

To visualize the difference in recoverability of MNIST and CIFAR-10,
```python
python3 compare_recovery.py
```

To visualize the averaged composite image from each dataset,
```python
python3 avg_images.py
```
