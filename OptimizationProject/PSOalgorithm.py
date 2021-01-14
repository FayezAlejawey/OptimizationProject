from __future__ import print_function
from multiprocessing.dummy import Pool as ThreadPool

import logging
import numpy as np

from scipy.integrate import ode

def halton_sequence(size, dim):
    seq = np.zeros( (dim,size) )
    primeGen = next_prime()
    next(primeGen)

    for d in range(dim):
        base = next(primeGen)
        seq[d] = np.array([vdc(i, base) for i in range(size)])

    return seq

def next_prime():
    def is_prime(num):
        "Checks if num is a prime value"
        for i in range(2,int(num**0.5)+1):
            if (num % i) == 0: return False
        return True

    prime = 3
    while(1):
        if is_prime(prime):
            yield prime
        prime += 2

def vdc(n, base=2):
    vdc, denom = 0,1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder / float(denom)
    return vdc

class Particle:
    def __init__(self, dim=10):
        self.__dim = dim

class PSO:

    logger = logging.getLogger(__name__)

    def __init__(self, func, bounds, initPos=None, nPart=None, **kwargs):
        """
        Performs Particle Swarm Optimisation to find the minimum of
        `func` function within provided parameters `bounds` range.
        The `func` should be a callable, and `bounds` a list-like
        where the first and second indices contain min and max bounds,
        respectively.
        Additionally one can set up `initPos` initial positions (see
        self.setInitPos) and the number `nPart` of used particles.
        """

        # Params
        self.epsError = 1.
        self.maxGen = 5000
        self.minRepetition = 100
        self.maxRepetition = 100

        self.w = 0.01
        self.phiP = 0.2  # Local best
        self.phiG = 0.1 # Global best
        self.phiN = 0.01 # Noise affect

        # Function to be minimised
        self.problem = func

        # Set up boundary values
        self.minBound = np.array(bounds[0])
        self.maxBound = np.array(bounds[1])

        self.dim = len(bounds[0])

        # Noise preparation
        self.nMean = np.zeros(self.dim)
        cov = np.sqrt(self.maxBound - self.minBound)/2
        self.nCov = np.diag(cov)
        MN = np.random.multivariate_normal
        cov[cov==0] = 1
        norm = np.sum(cov)*2*np.pi
        self.noise = lambda: (MN(self.nMean, self.nCov, 1)/norm).flatten()

        # Speed vector
        self.speed = np.ones(self.dim)
        # Number of particles in swarm
        if nPart is None:
            nPart = 100
        self.nPart = nPart

        # Initial positions
        if initPos is not None:
            initPos = np.array(initPos).reshape((-1,self.dim))
        self.initPos = initPos

        # How often print debug update
        self.refreshRate = 1 # %

        threads = 4 if "threads" not in kwargs else kwargs["threads"]
        self.pool = ThreadPool(threads)

    def setInitPos(self, initPos):
        """
        Variable `initPos` sets initial values for the PSO.
        It is a 2D list of (iParts, params) shape.
        If `iParts` is smaller than self.nPart then it will initiate
        first iParts with provided parameters and the rest particles
        will be assigned with random params within boundaries.
        """
        initPos = np.array(initPos).reshape((-1,self.dim))
        self.initPos = initPos

    def __initPart(self):
        """Initiate particles"""

        _size = initPos.shape[0] if self.initPos is not None else 0
        self.seq = halton_sequence(self.nPart-_size+3, self.dim)

        # Create particles
        self.Particles = [Particle(self.dim) for i in range(self.nPart)]

        # Position in generated semi-random sequence
        seqPos = 0

        # Initiate pos and fit for particles
        for part in self.Particles:

            # Initial position
            if self.initPos == None:

                part.pos = self.seq[:,seqPos]*(self.maxBound-self.minBound)
                part.pos += self.minBound
                seqPos += 1
            else:
                part.pos = self.initPos[0,:]
                self.initPos = np.delete(self.initPos, 0,0)

                # If nothing left on initial pos
                if len(self.initPos) == 0: self.initPos = None

            # Initial velocity
            part.vel = np.random.random(self.dim)*(np.sqrt(self.maxBound - self.minBound)/2)
            part.vel *= [-1., 1.][np.random.random()>0.5]

            # Initial fitness
            part.fitness = self.problem(part.pos)
            part.bestFit = part.fitness
            part.bestPos = part.pos

        # Global best fitness
        self.globBestFit = self.Particles[0].fitness
        self.globBestPos = self.Particles[0].pos
        for part in self.Particles:
            if part.fitness < self.globBestFit:
                self.globBestFit = part.fitness
                self.globBestPos = part.pos

    def update(self):
        results = self.pool.map(self._update_single, self.Particles)

        # Global and local best fitness
        for part in self.Particles:

            # Comparing to local best
            if part.fitness < part.bestFit:
                part.bestFit = part.fitness

            # Comparing to global best
            if part.fitness < self.globBestFit:
                self.globBestFit = part.fitness
                self.globBestPos = part.pos

    def _update_single(self, part):
        """Updates each step"""
        # Gen param
        rP, rG = np.random.random(2)
        # Replacing random values with speed vector

        w, phiP, phiG = self.w, self.phiP, self.phiG

        # Update velocity
        v, pos = part.vel, part.pos
        part.vel = self.w*v
        part.vel += phiP*rP*(part.bestPos-pos) # local best update
        part.vel += phiG*rG*(self.globBestPos-pos) # global best update

        part.vel += self.phiN*self.noise() # perturbation

        # New position
        part.pos += part.vel

        # If pos outside bounds
        if np.any(part.pos<self.minBound):
            idx = part.pos<self.minBound
            part.pos[idx] = self.minBound[idx]
        if np.any(part.pos>self.maxBound):
            idx = part.pos>self.maxBound
            part.pos[idx] = self.maxBound[idx]

        # New fitness
        part.fitness = self.problem(part.pos)

    def getGlobalBest(self):
        return self.globBestPos, self.globBestFit

    def optimize(self):
        """ Optimisation function.
            Before it is run, initial values should be set.
        """

        # Initiate particles
        self.__initPart()
        self.lastGlobBestFit = 0
        self.changeCounter = 0

        idx = 0
        while(idx < self.maxGen):
            if idx % int(self.maxGen*self.refreshRate*0.01) == 0:
                self.logger.debug("Gen: {}/{}  -- best = {}".format(idx, self.maxGen, self.globBestFit))

            # Perform search
            self.update()

            # Acceptably close to solution
            if self.globBestFit < self.epsError:
                return self.getGlobalBest()

            if self.globBestFit == self.lastGlobBestFit and \
                idx > self.minRepetition:

                self.changeCounter += 1
                if self.changeCounter == self.maxRepetition:
                    self.logger.debug("Obtained limit of repeating the same value.")
                    self.logger.debug("Stopping calculations.")
                    break
            else:
                self.changeCounter = 0
            self.lastGlobBestFit = self.globBestFit

            # next gen
            idx += 1

        # Search finished
        return self.getGlobalBest()

#################################
def PSOAlgorithm(x1, x2, y, isFirstOrderPoly):
    if __name__ == "__main__":
        logging.basicConfig(level=logging.DEBUG)

        rec = None
        if not x2:
            if isFirstOrderPoly:
                rec = lambda a: np.multiply(a[1], x1) + a[0]
            else:
                rec = lambda a: np.multiply(a[2], [item**2  for item in x1]) + np.multiply(a[1], x1) + a[0]
        else:
            rec = lambda a: np.multiply(a[2], [item**2  for item in x2]) + np.multiply(a[1], x1) + a[0]
        
        
        minProb = lambda a: np.sum(np.abs(y-rec(a))**2)

        numParam = 4
        bounds = ([0]*numParam, [10]*numParam)

        config = {"threads": 8}
        pso = PSO(minProb, bounds, **config)
        bestPos, bestFit = pso.optimize()

        print('bestFit: ', bestFit)
        print('bestPos: ', bestPos)

        ############################
        # Visual results representation
        try:
            import pylab as plt
        except ImportError:
            print()
            print("Skipping comparison plot as Matplotlib is not installed.")
            print("This is bonus, so don't worry, you're not missing anything.")
            import sys
            sys.exit()

        if not x2:
            if isFirstOrderPoly:
                plt.figure()
                plt.scatter(x1, y)
                plt.plot(x1, rec(bestPos), 'r')
                plt.xlabel("x1")
                plt.ylabel("y")
                plt.title("Approximating First Order Polynomial Using PSO Algorithm")
            else:
                plt.figure()
                plt.scatter(x1, y)
                plt.plot(x1, rec(bestPos), 'r')
                plt.xlabel("x1")
                plt.ylabel("y")
                plt.title("Approximating Second Order Polynomial Using PSO Algorithm")
        else:
            fig3D = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(x1, x2, y);
            ax.plot3D(x1, x2, rec(bestPos), color='red')
            ax.set_title("Approximating Two Dimensional First Order Polynomial Using PSO Algorithm")
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('y');

        #plt.savefig('fit',dpi=120)
        plt.show()

##############################
#Approximate a First Order Polynomial (y = a_1 * x + a_0) Using PSO Algorithm
PSOAlgorithm([0, 1, 2, 3, 4, 5], None, [2.1, 7.7, 13.6, 27.2, 40.9, 61.1], True)

#Approximate a Second Order Polynomial (y = a_2 * x^2 + a_1 * x + a_0) Using PSO Algorithm
PSOAlgorithm([0, 1, 2, 3, 4, 5], None, [2.1, 7.7, 13.6, 27.2, 40.9, 61.1], False)

#Approximate a Two Dimensional First Order Polynomial (y = a_2 * x_2 + a_1 * x_1 + a_0) Using PSO Algorithm
PSOAlgorithm([0, 2, 2.5, 1, 4, 7], [0, 1, 2, 3, 6, 2], [5, 10, 9, 0, 3, 27], False)