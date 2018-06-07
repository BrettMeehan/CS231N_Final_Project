import numpy as np
import matplotlib.pyplot as plt

roc = np.load('mobilenet_RGBI_16_pr.npy')
plt.plot(roc[:, 0], roc[:, 1])
plt.show()


