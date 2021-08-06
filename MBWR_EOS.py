import numpy as np

x1 = 0.8623085097507421
x2 = 2.976218765822098
x3 = -8.402230115796038
x4 = 0.1054136629203555
x5 = -0.8564583828174598
x6 = 1.582759470107601
x7 = 0.7639421948305453
x8 = 1.753173414312048
x9 = 2.798291772190376e+3
x10 = -4.8394220260857657e-2
x11 = 0.9963265197721935
x12 = -3.698000291272493e+1
x13 = 2.084012299434647e+1
x14 = 8.305402124717285e+1
x15 = -9.574799715203068e+2
x16 = -1.477746229234994e+2
x17 = 6.398607852471505e+1
x18 = 1.603993673294834e+1
x19 = 6.805916615864377e+1
x20 = -2.791293578795945e+3
x21 = -6.245128304568454
x22 = -8.116836104958410e+3
x23 = 1.488735559561229e+1
x24 = -1.059346754655084e+4
x25 = -1.131607632802822e+2
x26 = -8.867771540418822e+3
x27 = -3.986982844450543e+1
x28 = -4.689270299917261e+3
x29 = 2.593535277438717e+2
x30 = -2.694523589434903e+3
x31 = -7.218487631550215e+2
x32 = 1.721802063863269e+2


class MBWR_EOS():
    def __init__(self, fluid):
        self.T = fluid["temperature"]
        self.component = fluid["component"]
        self.sigma = np.array(fluid["sigma"])
        self.epsilon = np.array(fluid["epsilon"])
        self.wave = np.array(fluid["wave"])

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, rho):
        self._rho = np.array(rho)
        self.update()
        # print("update the density")

    def update(self):
        # vector version
        rho = (self._rho).reshape([1, -1])

        x = np.array(rho/rho.sum())
        self.X = x.T.dot(x)

        # vector version
        sigmaR = (self.sigma).reshape([1, -1]) # generate a row vector
        sigmaR = np.repeat(sigmaR, [self.component], axis=0)

        sigmaC = sigmaR.T

        self.sigma3 = ( (sigmaC + sigmaR) / 2 )**3


        # normal version
        # sigma3 = np.zeros((self.component,self.component))
        # for i in range(self.component):
            # for j in range(self.component):
                # sigma3[i,j] = ( (self.sigma[i] + self.sigma[j])/2.0 )**3


        # vector version
        epsilon = (self.epsilon).reshape([1,-1])
        self.epsilon3 = np.sqrt(epsilon.T.dot(epsilon))

        # normal version
        # epsilon3 = np.zeros((self.component,self.component))
        # for i in range(self.component):
            # for j in range(self.component):
                # epsilon3[i,j] = np.sqrt( self.epsilon[i] * self.epsilon[j] )

        self.sigmax = (self.X * self.sigma3).sum(axis=(0,1))

        self.epsilonx = 1/self.sigmax * (self.X * self.sigma3 * self.epsilon3).sum(axis=(0,1))

        self.Tstar = self.T / self.epsilonx

        self.rhostar = np.sum(self._rho) * self.sigmax
        # print("self.rhostar = {0}".format(self.rhostar))

        a = np.zeros((8,))
        Tstar = self.Tstar

        a[0] = x1*Tstar + x2 * np.sqrt(Tstar) + x3 + x4/Tstar + x5/Tstar**2
        a[1] = x6*Tstar + x7 + x8/Tstar + x9/Tstar**2
        a[2] = x10*Tstar + x11 + x12/Tstar
        a[3] = x13
        a[4] = x14/Tstar + x15/Tstar**2
        a[5] = x16/Tstar
        a[6] = x17/Tstar + x18/Tstar**2
        a[7] = x19/Tstar**2

        self.a = a

        b = np.zeros((6,))

        b[0] = x20/Tstar**2 + x21/Tstar**3
        b[1] = x22/Tstar**2 + x23/Tstar**4
        b[2] = x24/Tstar**2 + x25/Tstar**3
        b[3] = x26/Tstar**2 + x27/Tstar**4
        b[4] = x28/Tstar**2 + x29/Tstar**3
        b[5] = x30/Tstar**2 + x31/Tstar**3 + x32/Tstar**4
        
        self.b = b

        c = np.zeros((8,))

        c[0] = x2 * np.sqrt(Tstar)/2 + x3 + 2*x4/Tstar + 3*x5/Tstar**2
        c[1] = x7 + 2*x8/Tstar + 3*x9/Tstar**2
        c[2] = x11 + 2*x12/Tstar
        c[3] = x13
        c[4] = 2*x14/Tstar + 3*x15/Tstar**2
        c[5] = 2*x16/Tstar
        c[6] = 2*x17/Tstar + 3*x18/Tstar**2
        c[7] = 3*x19/Tstar**2

        self.c = c

        d = np.zeros((6,))
        
        d[0] = 3*x20/Tstar**2 + 4*x21/Tstar**3
        d[1] = 3*x22/Tstar**2 + 5*x23/Tstar**4
        d[2] = 3*x24/Tstar**2 + 4*x25/Tstar**3
        d[3] = 3*x26/Tstar**2 + 5*x27/Tstar**4
        d[4] = 3*x28/Tstar**2 + 4*x29/Tstar**3
        d[5] = 3*x30/Tstar**2 + 4*x31/Tstar**3 + 5*x32/Tstar**4

        self.d = d

        G = np.zeros((6,))
        gamma = 3
        F = np.exp(-gamma * self.rhostar**2)

        G[0] = (1-F)/(2*gamma)
        for i in range(1,6):
            G[i] = -(F*self.rhostar**(2*i) - (2*i)*G[i-1] )/(2*gamma)

        self.G = G


    def pressure(self):

        F = np.exp(-3.0*self.rhostar**2)
        pressure = self.rhostar * self.Tstar

        pressure += np.sum(self.a * [self.rhostar**(i+2) for i in range(8)])
        pressure += F * np.sum(self.b * [self.rhostar**(2*(i+1)+1) for i in range(6)])

        return pressure
    
    def exFreeEnergy(self):
        rho = np.sum(self._rho)

        A = 0
        A += np.sum(self.a * [self.rhostar**(i+1)/(i+1) for i in range(8)])
        A += np.sum(self.b * self.G)
        A *= self.epsilonx

        return A

    def exChemicalPotential(self):

        rho = np.sum(self._rho)

        A = 0
        A += np.sum(self.a * [self.rhostar**(i+1)/(i+1) for i in range(8)])
        A += np.sum(self.b * self.G)
        # print("A is equal to {}".format(A))


        U = 0
        U += np.sum(self.c * [self.rhostar**(i+1)/(i+1) for i in range(8)])
        U += np.sum(self.d * self.G)
        # print("U is equal to {}".format(U))

        dsigma = 2/rho * (np.dot(self._rho/rho, self.sigma3) - self.sigmax)

        depsilon = - self.epsilonx/self.sigmax * dsigma
        depsilon += 2/rho*( np.dot(self._rho/rho, (self.sigma3*self.epsilon3)) / self.sigmax - self.epsilonx)

        dA = (self.pressure()/(self.rhostar**2) - self.Tstar/self.rhostar) * (self.sigmax + rho*dsigma)
        dA -= (A - U)/self.epsilonx * (depsilon)

        chemicalPotential = A*self.epsilonx + A*rho*depsilon + rho*self.epsilonx*dA
        chemicalPotential /= self.T

        return chemicalPotential

    def chemicalPotential(self):

        chemP = np.log(self.rho * self.wave**3)
        chemP += self.exChemicalPotential()

        return chemP

    def getExFreeEnergy(self, density):

        self.rho = density

        return self.exFreeEnergy()


    def getExChemicalPotential(self, density):

        # print("get excess chemical potential")
        self.rho = density

        return self.exChemicalPotential()

    def getChemicalPotential(self, density):

        self.rho = density

        return self.chemicalPotential()



if __name__ == "__main__":
    test = {"temperature": 1.5, "sigma": [1, 1.1], "component": 2, "epsilon": [1.2, 1.]}
    mbwr = MBWR_EOS(test)
    mbwr.rho = [0.1, 0.2]
    pressure = mbwr.pressure()
    print("pressure is equal to " + str(pressure))
    mbwr.rho = [0.15, 0.15]

    # print(mbwr.getExChemicalPotential([0.1,0.15]))

    mbwr.rho = [0.0, 0.15]
    pressure = mbwr.pressure()
    print("pressure is equal to " + str(pressure))
    chemP = mbwr.exChemicalPotential()
    print("chemical potential is equal to " + str(chemP))
    print("chemical potential from eos is equal -0.893817898 and -0.8357211832")
    mbwr.rho = [0.1, 0.15]
    pressure = mbwr.pressure()
    # chemP = mbwr.chemicalPotential()
    # print("chemical potential is equal to " + str(chemP))
    # print("chemical potential from eos is equal -1.36958631 and -1.25968227441")

    test1 = {"temperature": 1.5, "sigma": [1.1, 1.1], "component": 2, "epsilon": [1.2, 1.]}
    mbwr = MBWR_EOS(test1)
    mbwr.rho = [0.12, 0.0]
    chemP = mbwr.exChemicalPotential()
    print("chemical potential is equal to " + str(chemP))
    print("chemical potential from eos is equal -1.01339")
