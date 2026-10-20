import umap
import sklearn
from sklearn.datasets import load_digits
from sklearn import datasets
import matplotlib.pyplot as plt

# dataset documentation: https://scikit-learn.org/1.5/auto_examples/datasets/plot_digits_last_image.html
X, y = load_digits(return_X_y=True)
# print(X.shape)

digits = datasets.load_digits()

# Display the last digit
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()