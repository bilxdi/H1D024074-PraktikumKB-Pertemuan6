# Import library
import numpy as np
import matplotlib.pyplot as plt

# Buat kelas Perceptron
class Perceptron:
# Simpan learning rate dan max epoch dalam konstruktor
    def __init__(self, alpha=0.1, epoch=10):
        self.alpha = alpha
        self.epoch = epoch