# image_completion
This is my final project submission for CS 539, Convex Optimization.
![Header Image](figs/fig_cmp.png?raw=true "Image Recovery")

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
