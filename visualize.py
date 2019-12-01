"""
Create visualizations of Keras history objects.
"""

import pickle
import matplotlib.pyplot as plt

WORK_PATH = './gen-task/'

def graph_history(history):
    plt.figure()
    plt.plot(history['loss'])
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.title('Loss for Generative Task')
    plt.show()

with open('{}history.pickle'.format(WORK_PATH), 'rb') as f:
    history = pickle.load(f)

    graph_history(history)