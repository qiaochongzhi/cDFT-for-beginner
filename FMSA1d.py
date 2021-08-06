import numpy as np
import scipy.integrate
import sys
import Functional

from scipy import signal

class FMSA1d(Functional.Functional):

    def __init__(self, fluid, system):

        super(FMSA1d, self).__init__(fluid, system)

        # ============ init DCF ============ #

        self.DCF = np.zeros((self.maxNum*2+1, self.fluid["component"], self.fluid["component"]))
        # print(self.DCF)

        eta = (np.pi/6) * np.sum(self.system["bulkDensity"]*self.fluid["sigma"]**3)

        # print(self.uattz(10, 0, 1, 1, 0.2)/10)

        for i in range(self.fluid["component"]):
            for j in range(i, self.fluid["component"]):

                sigma = (self.fluid["sigma"][i] + self.fluid["sigma"][j]) / 2
                epsilon = np.sqrt(self.fluid["epsilon"][i] * self.fluid["epsilon"][j]) \
                                  / self.fluid["temperature"]

                u = [scipy.integrate.quad(self.uattz,0,np.inf,args=(np.abs(z)*self.gridWidth,\
                                        epsilon,sigma,eta))[0] for z in range(-self.maxNum, self.maxNum+1)]

                u = - np.array(u)

                u *= 2 * np.pi
                if i == j:
                    self.DCF[:,i,j] = u
                else:
                    self.DCF[:,i,j] = u
                    self.DCF[:,j,i] = u

        self.DCF[0,:,:] = self.DCF[0,:,:]/2
        self.DCF[-1,:,:] = self.DCF[-1,:,:]/2

        # print(self.DCF.T)
        # print(np.sum(self.DCF))

    def uattPY(self, rr, epsilon, sigma, eta):

        Tstar = 1.0/epsilon
        d = (1+0.2977*Tstar)/(1 + 0.33163*Tstar + 0.0010477*Tstar**2) * sigma
        r = rr/d

        if r > 1:
            u = 0
        else:
            u = -eta*(1+2*eta)**2 *r**4 /( 2*(1-eta)**4 )
            u += 6*eta*(1+eta+eta**2/4)*r**2 / ( 1-eta )**4
            u -= (1+2*eta)**2 * r / ( (1-eta)**4 )
        return u/r

    def uattz(self, rr, z, epsilon, sigma, eta):
        r = np.abs( np.sqrt(rr**2 + z**2) )
        # ucutoff = 4*epsilon * ((sigma/cutoff)**12 - (sigma/cutoff)**6)
        # print(ucutoff)

        Tstar = 1.0/epsilon

        d = (1+0.2977*Tstar)/(1 + 0.33163*Tstar + 0.0010477*Tstar**2) * sigma

        k0 = 2.1714  * sigma
        z1 = 2.9637  / sigma
        z2 = 14.0167 / sigma

        k1 = k0 * np.exp(z1 * (sigma-d))
        k2 = k0 * np.exp(z2 * (sigma-d))

        Tstar1 = Tstar * d / k1
        Tstar2 = Tstar * d / k2

        u = self.catt(eta, Tstar1, z1*d, r/d) - self.catt(eta, Tstar2, z2*d, r/d)

        return u * rr

    def catt(self, eta, Tstar, z, r):

        S = lambda t: ((1-eta)**2) * t**3 + 6*eta*(1-eta)*(t**2) + 18*(eta**2)*t - 12*eta*(1+2*eta)
        L = lambda t: (1+eta/2)*t + 1 + 2*eta
        Q = lambda t: ( S(t) + 12*eta*L(t)*np.exp(-t) )/( ((1-eta)**2) * (t**3) )

        if r > 1:
            c = np.exp(-z*(r-1))/Tstar
        else:
            c = (S(z)**2) * np.exp(-z*(r-1)) + 144*(eta**2) * (L(z)**2) * np.exp(z*(r-1))
            c -= 12*(eta**2) * ( ((1+2*eta)**2)*(z**4) + (1-eta)*(1+2*eta)*(z**5) )*(r**4)
            c += 12*eta*( S(z)*L(z)*(z**2) - ((1-eta)**2)*(1+eta/2)*(z**6) )*(r**2)
            c -= 24*eta*( ((1+2*eta)**2)*(z**4) + (1-eta)*(1+2*eta)*(z**5) )*r
            c += 24*eta*S(z)*L(z)
            c /= -( ((1-eta)**4) * (z**6) * (Q(z)**2) * Tstar )
            c += np.exp(-z*(r-1))/Tstar

        return c/r

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, density):
        self._density = density

    '''
    [old version] In this version, I create a matrix to achieve the convolution.
    '''
    # def exChemicalPotential(self):

        # density = self._density - self.system["bulkDensity"].reshape((self.fluid["component"], -1))
        # densityMatrix = self.densityIntegrate(density, self.fluid["component"], self.maxNum, self.system["grid"])

        # exChemP = np.zeros((self.fluid["component"], self.system["grid"]))
        # for i in range(self.fluid["component"]):

            # x = np.sum(densityMatrix * self.DCF[:,:,i].reshape((-1,self.fluid["component"],1)), axis = 0)
            # exChemP[i, :] = np.sum(x, axis = 0)

        # exChemP *= self.gridWidth

        # return exChemP

    '''
    In this version, I achieve the convolution by using FFT (from scipy)
    '''
    def exChemicalPotential(self):

        density = self._density - self.system["bulkDensity"].reshape((self.fluid["component"], -1))

        return self.densityIntegrateFFT(density, self.DCF, self.fluid["component"], self.maxNum, self.system["grid"])


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import MBWR_EOS
    import FMT1d

    fluid = {}
    fluid["type"] = "LJ"
    fluid["component"] = 2
    fluid["sigma"] = np.array([1.0, 1.0])
    fluid["epsilon"] = np.array([1.0, 1.0])
    fluid["diameter"] = np.array([1.0, 1.0])
    fluid["temperature"] = 1

    system = {}
    system["grid"] = 600
    system["bulkDensity"] = np.array([0.2, 0.2])
    system["boundaryCondition"] = 1
    system["size"] = 30
    system["cutoff"] = np.array([6])

    # testFMT = FMT1d.FMT1d(fluid, system)
    testFMSA = FMSA1d(fluid, system)
    testFMSA.density = np.zeros((fluid["component"], system["grid"])) + system["bulkDensity"].reshape((fluid["component"], -1))
    # print(testFMSA.exChemicalPotential())

    x = [[x, testFMSA.uattz(x, 0, 1/1.5, 1, 0.4*np.pi/6)/x] for x in np.linspace(0.01,3,600)]
    x = np.array(x)

    y = [[0, testFMSA.uattPY(x, 1/1.5, 1, 0.4*np.pi/6)] for x in np.linspace(0.01,3,600)]
    y = np.array(y)

    z = x + y

    y = np.loadtxt("./Comparison/LJ_cDFT/cr_att.dat")
    plt.figure()
    plt.xlim((0,2.0))
    plt.ylim((-5,1.2))
    plt.plot(z[:,0], z[:,1])
    plt.scatter(y[:,0],y[:,1])
    plt.savefig("./Comparison/LJ_cDFT/cr_att_FMSA.jpg")
    # plt.show()


