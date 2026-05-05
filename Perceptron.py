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