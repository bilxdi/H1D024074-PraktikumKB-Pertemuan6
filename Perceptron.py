# Import library
import numpy as np
import matplotlib.pyplot as plt

# Buat kelas Perceptron
class Perceptron:
# Simpan learning rate dan max epoch dalam konstruktor
    def __init__(self, alpha=0.1, epoch=10):
        self.alpha = alpha
        self.epoch = epoch

    # Fungsi menghitung nilai y_in atau net
    def weighted_sum(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    # Fungsi menerapkan fungsi aktivasi bipolar
    def predict(self, X):
        return np.where(self.weighted_sum(X) >= 0.0, 1, -1)
    
    # Fungsi membuat simulasi garis pemisah data
    def plot_decision_boundary(self, X, t, epoch):
        # Membuat titik data input
        plt.scatter(X[:, 0], X[:, 1], c=t.ravel(), marker='o',
        edgecolors='k', cmap=plt.cm.RdYlBu)

        # Menentukan limit tampilan bidang grafik
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        # Membuat garis pemisah
        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = -(self.w_[0] + self.w_[1] * x_vals) / self.w_[2]
        plt.plot(x_vals, y_vals, 'b-', label=f'Decision boundary (Epoch {epoch+1})')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(f"Decision Boundary Pada Epoch {epoch+1}")
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.show()
