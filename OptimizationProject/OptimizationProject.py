#Import Needed Python Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 9)

#One-Dimensional Probelm Input Data
x = [0, 1, 2, 3, 4, 5]
y = [2.1, 7.7, 13.6, 27.2, 40.9, 61.1]

#Building The Model Using Steepest Descent Algorithm To Approximate a First Order Polynomial (y = a_1 * x + a_0)
a_0 = 1
a_1 = 1

learningRate = 0.0001
lengthOfInputData = float(len(x))

while True:
    yPredicted = [a_1*x[i] + a_0 for i in range(0, len(x))]#a_1*x + a_0Lst

    yDiff = [y[i] - yPredicted[i] for i in range(0, len(x))]
    
    derivativeWithRespectToa_1 = (-2/lengthOfInputData) * sum([x[i] * yDiff[i] for i in range(0, len(x))])
    derivativeWithRespectToa_0 = (-2/lengthOfInputData) * sum(yDiff)

    a_1Old = a_1
    
    a_1 = a_1 - learningRate*derivativeWithRespectToa_1
    a_0 = a_0 - learningRate*derivativeWithRespectToa_0

    diff = abs(a_1Old - a_1)
    if diff < 0.000001:
        break

#Plot The Scattered (x, y) & The Predicted Line (x, yPredicted)
yPredicted = [a_1*x[i] + a_0 for i in range(0, len(x))]

plt.scatter(x, y)
plt.plot([min(x), max(x)], [min(yPredicted), max(yPredicted)], color = 'red')
plt.suptitle('Approximating First Order Polynomial Using Steepest Descent Algorithm', fontsize=14, fontweight='bold')
plt.xlabel('x')
plt.ylabel('y')
plt.show()