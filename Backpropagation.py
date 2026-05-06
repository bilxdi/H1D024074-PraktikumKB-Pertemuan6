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

        # Fungsi menerapkan fungsi aktivasi sigmoid bipolar atau tanh
        def bi_sigmoid(self, x):
            return np.tanh(x)
        
        # Fungsi menerapkan turunan fungsi aktivasi sigmoid bipolar atau tanh (asumsi x = output sigmoid bipolar/tanh)
        def deriv_bi_sigmoid(self, x):
            return 1 - x**2

        # Fungsi membuat simulasi perbaikan bobot dan bias
        def plot_error(self, x, epoch):
            plt.plot(range(1, epoch + 1), x, linestyle='-', color='b', label='Error')

            final_error = x[-1]
            plt.annotate(f'Epoch {epoch}, Error: {final_error:.4f}', xy=(epoch, final_error), xytext=(epoch - len(x) * 0.2, final_error + 0.05), arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=10, color='red')
            
            plt.title('Perbaikan Error Setiap Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Sum Square Error(SSE)')
            plt.grid(True)
            plt.legend()
            plt.show()