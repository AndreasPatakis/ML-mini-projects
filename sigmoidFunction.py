
# Import matplotlib, numpy and math
import matplotlib.pyplot as plt
import numpy as np
import math

x = np.linspace(-10, 10, 100)

z = 1/(1 + np.exp(-(1*x)))

plt.figure(1)
plt.plot(x, z)
plt.xlabel("x")
plt.ylabel("Sigmoid(X)")


plt.figure(2)
plt.scatter(5,7)


plt.show()
