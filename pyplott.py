from matplotlib import pyplot as plt
import numpy as np

aut = np.random.rand(3,4,12)
fit = np.random.rand(3,4)

fig, (ax1, ax2) = plt.subplots(1,2)
ax2.matshow(fit,cmap=plt.cm.Blues)
fit = np.random.rand(3,4)
plt.show(block=False)
plt.pause(5)
ax2.clear()
ax2.matshow(fit,cmap=plt.cm.Blues)
plt.show(block=False)
plt.pause(5)
