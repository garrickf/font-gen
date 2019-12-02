"""
Create visualizations of Keras history objects.
"""

import pickle
import matplotlib.pyplot as plt

# Edit WORK_PATH and FILES to desired
GEN_WORK_PATH = './gen-task/'
GEN_TASK_DATA = [
    ('fonts-all-2908_exp0_d2019-11-30_23h-53m_history', 'Experiment 0'),
    ('fonts-all-2908_exp1_d2019-12-01_0h-52m_history', 'Experiment 1 (no conv)'),
    ('fonts-all-2908_exp2_d2019-12-01_2h-3m_history', 'Experiment 2 (more neurons)'),
    ('fonts-all-2908_exp3_d2019-12-01_4h-31m_history', 'Experiment 3 (more layers i)'),
    ('fonts-all-2908_exp4_d2019-12-01_11h-8m_history', 'Experiment 4 (more layers ii)'),
    ('fonts-all-2908_exp5_d2019-12-01_13h-5m_history', 'Experiment 5 (more layers and more neurons)')
]

INF_WORK_PATH = './infer-task/'
INF_TASK_DATA = [
    ('fonts-jpn-all_exp0_d2019-12-01_20h-15m_history', 'Experiment 0 (no transfer)'),
    ('fonts-jpn-all_exp1_d2019-12-01_20h-25m_history', 'Experiment 1 (transfer)'),
]

def graph_losses(histories, title):
    plt.figure()
    for history, label in histories:
        plt.plot(history['loss'], label=label)
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.title(title)
    plt.legend()
    plt.show()

"""
Generation (Latin) task
"""
histories = []
for filename, label in GEN_TASK_DATA:
    with open('{}{}.pickle'.format(GEN_WORK_PATH, filename), 'rb') as f:
        histories.append((pickle.load(f), label)) # Add history obj to list

graph_losses(histories, 'Loss for Generative Task')

"""
Inference (hiragana) task
"""
histories = []
for filename, label in INF_TASK_DATA:
    with open('{}{}.pickle'.format(INF_WORK_PATH, filename), 'rb') as f:
        histories.append((pickle.load(f), label)) # Add history obj to list

graph_losses(histories, 'Loss for Inference Task')
