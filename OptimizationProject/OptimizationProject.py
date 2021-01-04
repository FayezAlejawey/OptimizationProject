#Import Needed Python Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2)

fig3D = plt.figure()
ax = plt.axes(projection='3d')

def ApproximateFirstOrderPolynomialUsingSteepestDescentAlgorithm(x, y):
    #Building The Model Using Steepest Descent Algorithm To Approximate a First Order Polynomial (y = a_1 * x + a_0)
    a_0 = 1
    a_1 = 1

    learningRate = 0.0001
    lengthOfInputData = float(len(x))

    while True:
        yPredicted = [a_1*x[i] + a_0 for i in range(0, len(x))]

        yDiff = [y[i] - yPredicted[i] for i in range(0, len(x))]
    
        derivativeWithRespectToa_1 = (-2/lengthOfInputData) * sum([x[i] * yDiff[i] for i in range(0, len(x))])
        derivativeWithRespectToa_0 = (-2/lengthOfInputData) * sum(yDiff)

        a_1Old = a_1
        a_0Old = a_0
    
        a_1 = a_1 - learningRate*derivativeWithRespectToa_1
        a_0 = a_0 - learningRate*derivativeWithRespectToa_0

        diff = ((a_1Old - a_1)**2 + (a_0Old - a_0)**2)**0.5
        if diff < 0.000001:
            break

    #Plot The Scattered (x, y) & The Predicted Line (x, yPredicted)
    yPredicted = [a_1*x[i] + a_0 for i in range(0, len(x))]

    axs[0, 0].scatter(x, y)
    axs[0, 0].plot([min(x), max(x)], [min(yPredicted), max(yPredicted)], color = 'red')
    axs[0, 0].set_title('Approximating First Order Polynomial Using Steepest Descent Algorithm')

def ApproximateFirstOrderPolynomialUsingLevenbergMarquardtAlgorithm(x, y):
    #Building The Model Using Levenberg Marquardt Algorithm To Approximate a First Order Polynomial (y = a_1 * x + a_0)
    a_0 = 1
    a_1 = 1

    learningRate = 0.01
    beta = 10
    lengthOfInputData = float(len(x))

    oldDiff = None
    while True:
        yPredicted = [a_1*x[i] + a_0 for i in range(0, len(x))]

        yDiff = [y[i] - yPredicted[i] for i in range(0, len(x))]
    
        derivativeWithRespectToa_1 = (-2/lengthOfInputData) * sum([x[i] * yDiff[i] for i in range(0, len(x))])
        derivativeWithRespectToa_0 = (-2/lengthOfInputData) * sum(yDiff)

        a_1Old = a_1
        a_0Old = a_0

        a_1 = a_1 - (1/(2*learningRate))*derivativeWithRespectToa_1
        a_0 = a_0 - (1/(2*learningRate))*derivativeWithRespectToa_0

        diff = ((a_1Old - a_1)**2 + (a_0Old - a_0)**2)**0.5

        if oldDiff:
            if oldDiff < diff:
                learningRate = learningRate*beta

        oldDiff = diff

        if diff < 0.000001:
            break

    #Plot The Scattered (x, y) & The Predicted Line (x, yPredicted)
    yPredicted = [a_1*x[i] + a_0 for i in range(0, len(x))]

    axs[0, 1].scatter(x, y)
    axs[0, 1].plot([min(x), max(x)], [min(yPredicted), max(yPredicted)], color = 'red')
    axs[0, 1].set_title('Approximating First Order Polynomial Using Levenberg Marquardt Algorithm')

def ApproximateSecondOrderPolynomialUsingLevenbergMarquardtAlgorithm(x, y):
    #Building The Model Using Levenberg Marquardt Algorithm To Approximate a Second Order Polynomial (y = a_2 * x^2 + a_1 * x + a_0)
    a_0 = 1
    a_1 = 1
    a_2 = 1

    learningRate = 0.01
    beta = 10
    lengthOfInputData = float(len(x))

    oldDiff = None
    while True:
        yPredicted = [a_2*(x[i]**2) + a_1*x[i] + a_0 for i in range(0, len(x))]

        yDiff = [y[i] - yPredicted[i] for i in range(0, len(x))]
    
        derivativeWithRespectToa_2 = (-2/lengthOfInputData) * sum([(x[i]**2) * yDiff[i] for i in range(0, len(x))])
        derivativeWithRespectToa_1 = (-2/lengthOfInputData) * sum([x[i] * yDiff[i] for i in range(0, len(x))])
        derivativeWithRespectToa_0 = (-2/lengthOfInputData) * sum(yDiff)

        a_2Old = a_2
        a_1Old = a_1
        a_0Old = a_0

        a_2 = a_2 - (1/(2*learningRate))*derivativeWithRespectToa_2
        a_1 = a_1 - (1/(2*learningRate))*derivativeWithRespectToa_1
        a_0 = a_0 - (1/(2*learningRate))*derivativeWithRespectToa_0

        diff = ((a_2Old - a_2)**2 + (a_1Old - a_1)**2 + (a_0Old - a_0)**2)**0.5

        if oldDiff:
            if oldDiff < diff:
                learningRate = learningRate*beta

        oldDiff = diff

        if diff < 0.000001:
            break

    #Plot The Scattered (x, y) & The Predicted Line (x, yPredicted)
    yPredicted = [a_2*(x[i]**2) + a_1*x[i] + a_0 for i in range(0, len(x))]

    axs[1, 0].scatter(x, y)
    axs[1, 0].plot(x, yPredicted, color = 'red')
    axs[1, 0].set_title('Approximating Second Order Polynomial Using Levenberg Marquardt Algorithm')
 
def ApproximateTwoDimensionalFirstOrderPolynomialUsingLevenbergMarquardtAlgorithm(x1, x2, y):
    #Building The Model Using Levenberg Marquardt Algorithm To Approximate a Two Dimensional First Order Polynomial (y = a_2 * x_2 + a_1 * x_1 + a_0)
    a_0 = 1
    a_1 = 1
    a_2 = 1

    learningRate = 0.01
    beta = 10
    lengthOfInputData = float(len(x1))

    oldDiff = None
    while True:
        yPredicted = [a_2*x2[i] + a_1*x1[i] + a_0 for i in range(0, len(x1))]

        yDiff = [y[i] - yPredicted[i] for i in range(0, len(x1))]
    
        derivativeWithRespectToa_2 = (-2/lengthOfInputData) * sum([x2[i] * yDiff[i] for i in range(0, len(x1))])
        derivativeWithRespectToa_1 = (-2/lengthOfInputData) * sum([x1[i] * yDiff[i] for i in range(0, len(x1))])
        derivativeWithRespectToa_0 = (-2/lengthOfInputData) * sum(yDiff)

        a_2Old = a_2
        a_1Old = a_1
        a_0Old = a_0

        a_2 = a_2 - (1/(2*learningRate))*derivativeWithRespectToa_2
        a_1 = a_1 - (1/(2*learningRate))*derivativeWithRespectToa_1
        a_0 = a_0 - (1/(2*learningRate))*derivativeWithRespectToa_0

        diff = ((a_2Old - a_2)**2 + (a_1Old - a_1)**2 + (a_0Old - a_0)**2)**0.5

        if oldDiff:
            if oldDiff < diff:
                learningRate = learningRate*beta

        oldDiff = diff

        if diff < 0.000001:
            break

    #Plot The Scattered (x, y) & The Predicted Line (x, yPredicted)
    yPredicted = [a_2*x2[i] + a_1*x1[i] + a_0 for i in range(0, len(x1))]

    ax.scatter3D(x1, x2, y);
    ax.plot3D(x1, x2, yPredicted, color='red')
    ax.set_title('Approximating Two Dimensional First Order Polynomial Using Levenberg Marquardt Algorithm')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y');

#One-Dimensional First-Order Polynomial Problem Input Data
ApproximateFirstOrderPolynomialUsingSteepestDescentAlgorithm([0, 1, 2, 3, 4, 5], [2.1, 7.7, 13.6, 27.2, 40.9, 61.1])
ApproximateFirstOrderPolynomialUsingLevenbergMarquardtAlgorithm([0, 1, 2, 3, 4, 5], [2.1, 7.7, 13.6, 27.2, 40.9, 61.1])

#One-Dimensional Second-Order Polynomial Problem Input Data
ApproximateSecondOrderPolynomialUsingLevenbergMarquardtAlgorithm([0, 1, 2, 3, 4, 5], [2.1, 7.7, 13.6, 27.2, 40.9, 61.1])

#Two-Dimensional First-Order Polynomial Problem Input Data
ApproximateTwoDimensionalFirstOrderPolynomialUsingLevenbergMarquardtAlgorithm([0, 2, 2.5, 1, 4, 7], [0, 1, 2, 3, 6, 2], [5, 10, 9, 0, 3, 27])

#Showing The Predefined Plots
plt.get_current_fig_manager().full_screen_toggle()
plt.show()