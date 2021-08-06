import numpy as np
import scipy.integrate
import sys
import Functional

from scipy import signal

class MFA1d(Functional.Functional):

    def __init__(self, fluid, system):

        super(MFA1d, self).__init__(fluid, system)

        # ============ init DCF ============ #

        self.DCF = np.zeros((self.maxNum*2+1, self.fluid["component"], self.fluid["component"]))
        # print(self.DCF)

        for i in range(self.fluid["component"]):
            for j in range(i, self.fluid["component"]):

                sigma = (self.fluid["sigma"][i] + self.fluid["sigma"][j]) / 2
                epsilon = np.sqrt(self.fluid["epsilon"][i] * self.fluid["epsilon"][j])/ self.fluid["temperature"]

                u = [scipy.integrate.quad(self.uattz,0,np.inf,args=(z*self.gridWidth,epsilon,sigma, \
                    self.system["cutoff"]))[0] for z in range(-self.maxNum, self.maxNum+1)]
                u = np.array(u)
                u *= 2 * np.pi

                if i == j:
                    self.DCF[:,i,j] = u
                else:
                    self.DCF[:,i,j] = u
                    self.DCF[:,j,i] = u

        # print(self.DCF.T)
        # print(np.sum(self.DCF))

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, density):
        self._density = density

    def uattPY(self, rr, epsilon, sigma, eta):

        Tstar = 1.0/epsilon
        d = (1+0.2977*Tstar) / (1 + 0.33163*Tstar + 0.0010477*Tstar**2) * sigma
        r = rr/d

        if r > 1:
            u = 0
        else:
            u = -eta*(1+2*eta)**2 * r**4 / ( 2*(1-eta)**4 )
            u += 6*eta*(1+eta+eta**2/4) * r**2 / ( 1-eta )**4
            u -= (1+2*eta)**2 * r / ( (1-eta)**4 )
        return u/r

    def uattz(self, rr, z, epsilon, sigma, cutoff):
        r = np.sqrt(rr**2 + z**2)
        ucutoff = 4*epsilon * ((sigma/cutoff)**12 - (sigma/cutoff)**6)
        # print(ucutoff)
        if r < (2**(1/6)) * sigma:
            u = -epsilon - ucutoff
        # if r < sigma:
            # u = 0
        elif r < cutoff:
            u = 4*epsilon * ((sigma/r)**12 - (sigma/r)**6) - ucutoff
        else:
            u = 0
        return u * rr

    '''
    [old version] In this version, I create a matrix to achieve the convolution.
    '''
    # def exChemicalPotential(self):

        # densityMatrix = self.densityIntegrate(self._density, self.fluid["component"], self.maxNum, self.system["grid"])

        # exChemP = np.zeros((self.fluid["component"], self.system["grid"]))
        # for i in range(self.fluid["component"]):

            # x = np.sum(densityMatrix * self.DCF[:,:,i].reshape((-1,self.fluid["component"],1)), axis = 0)
            # exChemP[i, :] = np.sum(x, axis = 0)

        # exChemP *= self.gridWidth

        # return exChemP

    '''
    In this version, I finished the convolution by using FFT (from scipy)
    '''
    def exChemicalPotential(self):

        # density = self._density
        if self.system["Cor"] == True:
            density = self._density
        else:
            density = self._density - self.system["bulkDensity"].reshape((self.fluid["component"], -1))

        return self.densityIntegrateFFT(density, self.DCF, self.fluid["component"], self.maxNum, self.system["grid"])



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import MBWR_EOS
    import FMT1d

    fluid = {}
    fluid["type"] = "LJ"
    fluid["component"] = 1
    fluid["sigma"] = np.array([1.0])
    fluid["epsilon"] = np.array([1.0])
    fluid["diameter"] = np.array([1.0])
    fluid["temperature"] = 1

    system = {}
    system["grid"] = 600
    system["bulkDensity"] = np.array([0.2])
    system["boundaryCondition"] = 1
    system["size"] = 30
    system["cutoff"] = np.array([6])

    # testFMT = FMT1d.FMT1d(fluid, system)
    testMFA = MFA1d(fluid, system)
    testMFA.density = np.zeros((fluid["component"], system["grid"])) + system["bulkDensity"]
    # print(testMFA.exChemicalPotential())

    x = [[x, -testMFA.uattz(x, 0, 1/1.5, 1, 5)/x] for x in np.linspace(0.001, 3, 600)]
    x = np.array(x)

    y = [[0, testMFA.uattPY(x, 1/1.5, 1, 0.4*np.pi/6)] for x in np.linspace(0.001, 3, 600)]
    y = np.array(y)

    z = x + y

    y = np.loadtxt("./Comparison/LJ_cDFT/cr_att.dat")
    plt.figure()
    plt.xlim((0,2.0))
    plt.ylim((-5,1.2))
    plt.plot(z[:,0], z[:,1])
    plt.scatter(y[:,0], y[:,1])
    plt.savefig("./Comparison/LJ_cDFT/cr_att_MFA.jpg")
