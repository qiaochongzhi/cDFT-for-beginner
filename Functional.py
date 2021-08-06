import numpy as np
import sys

from scipy import signal

from numba import jit

class Functional():

    def __init__(self, fluid, system):

        self.fluid = fluid
        self.system = system

        self._density  = np.zeros((self.fluid["component"], self.system["grid"]))
        self.gridWidth = self.system["size"] / self.system["grid"]

        if self.fluid["type"] == "LJ":
            num1        = self.system["cutoff"] / self.gridWidth
            self.num    = [int(round(x)) for x in num1]
            self.maxNum = np.max(self.num)
        elif self.fluid["type"] == "ion":
            num1        = self.fluid["diameter"] / self.gridWidth
            self.num    = [int(round(x)) for x in num1]
            self.maxNum = np.max(self.num)


    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, density):
        self._density = density

    def exChemicalPotential():
        pass

    def getExChemicalPotential(self, density):

        self.density = density
        exChemP = self.exChemicalPotential()
        return exChemP

    # @jit()
    def densityIntegrate(self, density, component, num, grid):
        densityMatrix = np.zeros((num*2 + 1, component, grid))
        densityMatrix[num,:,:] = density.copy()

        if self.system["boundaryCondition"] == "slitPore":
            # zerosMatrixL = np.zeros((component, i))
            zerosMatrixL = np.zeros((component, grid))
            zerosMatrixR = zerosMatrixL
        elif self.system["boundaryCondition"] == "singleWall":
            # zerosMatrixL = np.zeros((component, i))
            # zerosMatrixR = np.zeros((component, i))
            # zerosMatrixR += density[:,-1].reshape((component, -1))
            zerosMatrixL  = np.zeros((component, grid))
            zerosMatrixR  = np.zeros((component, grid)) 
            zerosMatrixR += density[:,-1].reshape((component, -1)) 
        elif self.system["boundaryCondition"] == "Bulk":
            # zerosMatrixL = density[:, -i:]
            # zerosMatrixR = density[:, :i]
            zerosMatrixL = density
            zerosMatrixR = zerosMatrixL
        else:
            print("wrong, densityIntegrate")
            sys.exit(-1)

        for i in range(1, num+1):

            matrix1 = np.hstack((zerosMatrixL[:, -i:], density[:, :-i]))
            matrix2 = np.hstack((density[:,i:], zerosMatrixR[:, :i]))

            densityMatrix[num-i,:,:] = matrix1
            densityMatrix[num+i,:,:] = matrix2

        return densityMatrix

        # matrix = testV(density, densityMatrix, zerosMatrixL, zerosMatrixR, num)
        # return matrix


    def densityIntegrateFFT(self, density, DCF, component, num, grid):

        if self.system["boundaryCondition"] == "singleWall":
            density1 = np.zeros((component, grid+num))
            density1[:, :grid] = density
            # density1[:, grid:] += self.system["bulkDensity"].reshape((component, -1))
            density1[:, grid:] += density[:,-1].reshape((component, -1))
        elif self.system["boundaryCondition"] == "slitPore":
            density1 = density
        else:
            print("Wrong, densityIntegrateFFT")
            sys.exit(-1)

        exChemP  = np.zeros_like(density1)
        exChemP1 = np.zeros_like(density1)

        if len(density1[0,:]) < 502:
            for i in range(component):
                for j in range(component):

                    exChemP1[j,:] = signal.convolve(density1[j,:], DCF[:,j,i][::-1], mode="same")
                    # exChemP1[j,:] = np.convolve(density1[j,:], DCF[:,j,i][::-1], mode="same")

                exChemP[i,:] = np.sum(exChemP1, axis = 0)
        else:
            for i in range(component):
                for j in range(component):

                    exChemP1[j,:] = signal.fftconvolve(density1[j,:], DCF[:,j,i][::-1], mode="same")
                    # exChemP1[j,:] = np.convolve(density1[j,:], DCF[:,j,i][::-1], mode="same")

                exChemP[i,:] = np.sum(exChemP1, axis = 0)

        exChemP *= self.gridWidth
        return exChemP[:,:grid]


# @jit(nopython=True, parallel=True)
# def testV(density, densityMatrix, zerosMatrixL, zerosMatrixR, num):

        # for i in range(1, num+1):

            # densityMatrix[num-i,:, :i] = zerosMatrixL[:, -i:]
            # densityMatrix[num-i,:, i:] = density[:, :-i]
            # densityMatrix[num+i,:, :-i] = density[:, i:]
            # densityMatrix[num+i,:, -i:] = zerosMatrixR[:, :i]

        # return densityMatrix



