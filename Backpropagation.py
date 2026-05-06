# Import library
import numpy as np
import matplotlib.pyplot as plt

# Buat kelas Backpropagation
class Backpropagation:
    # Simpan learning rate, epoch, dan target error dalam konstruktor serta inisialisasi bobot dan bias awal random
    def __init__(self, alpha, epoch, target_error):
        self.alpha = alpha
        self.epoch = epoch
        self.target_error = target_error

        self.n_input = 2
        self.n_hidden = 2
        self.n_output = 1

        self.w_hidden = np.random.rand(self.n_input, self.n_hidden)
        self.b_hidden = np.random.rand(1, self.n_hidden)
        
        self.w_output = np.random.rand(self.n_hidden, self.n_output)
        self.b_output = np.random.rand(1, self.n_output)
