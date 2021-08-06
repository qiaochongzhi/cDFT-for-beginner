import numpy as np

class BMCSL():

    def __init__(self, fluid):
        self.component = fluid["component"]
        self.diameter = np.array(fluid["diameter"])
        self.wave = np.array(fluid["wave"])

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, density):
        self._density = np.array(density)
        self.calculateXi()
        # print("renew density")


    def calculateXi(self):
        """
        calculate xi of fluid

        input: fluid (a dictionary which contains all property of fluid)
        retruns: xi (a numpy list) 
        """

        # normal version
        # xi = np.zeros((4,))

        # for i in range(4):
            # for j in range(fluid["component"]):
                # xi[i] += fluid["density"][j] * fluid["diameter"][j]**i

            # xi[i] *= np.pi/6

        # vector version
        xi = np.zeros((4,))

        diameter = np.array([(self.diameter)**i for i in range(4)])

        self.xi = (self._density * diameter).sum(axis=1) * np.pi/6


    def compressibility(self):
        """
        Calculate the compressibility by using BMCSL equation of state

        input: fluid (a dictionary which contains all property of fluid)
        retruns: the compressibility of fluid 
        """

        xi = self.xi

        P = xi[0] / (1-xi[3])
        P += 3*xi[1]*xi[2] / (1-xi[3])**2
        P += (3-xi[3]) * xi[2]**3 / (1-xi[3])**3
        return P / xi[0]

    def exChemicalPotential(self):
        """
        Calculate the chemical potential by using BMCSL equation of state

        input: fluid (a dictionary which contains all property of fluid)
        retruns: the chemical potential of fluid (a list)
        """
        mu = np.zeros(self.component)

        # normal version
        # for k in range(fluid["component"]):
            # d = fluid["diameter"][k]
            # mu[k] = np.log(fluid["density"][k] * fluid["wave"][k]**3)
            # mu[k] -= np.log( 1-xi[3] )
            # mu[k] += 3 * (xi[2]*d + xi[1]*d**2) / (1-xi[3])
            # mu[k] += 9 * xi[2]**2 * d**2 / (2 * (1-xi[3])**2)
            # mu[k] += compressibility(fluid) * xi[0] * d**3
            # mu1 = np.log(1-xi[3]) + xi[3]/(1-xi[3]) - xi[3]**2/(2*(1-xi[3])**2)
            # mu1 *= 3 * (xi[2] * d / xi[3])**2

            # mu2 = 2*np.log(1-xi[3]) + xi[3]*(2-xi[3])/(1-xi[3])
            # mu2 *= -(xi[2]*d/xi[3])**3

            # mu[k] += mu1 + mu2

        # vector version
        diameter = self.diameter
        density = self._density
        xi = self.xi

        # mu = np.log(density * self.wave**3)
        if xi[3] < 1e-4:
            mu = np.zeros((self.component))
        elif (1 - xi[3]) < 1e-5:
            mu = -np.log( 1e-5 )
        else:
            mu = -np.log( 1-xi[3] )
            mu += 3 * (xi[2]*diameter + xi[1]*diameter**2) / (1-xi[3])
            mu += 9 * xi[2]**2 * diameter**2 / (2 * (1-xi[3])**2)
            mu += self.compressibility() * xi[0] * diameter**3
            mu1 = np.log(1-xi[3]) + xi[3]/(1-xi[3]) - xi[3]**2/(2*(1-xi[3])**2)
            mu1 *= 3 * (xi[2] * diameter / xi[3])**2

            mu2 = 2*np.log(1-xi[3]) + xi[3]*(2-xi[3])/(1-xi[3])
            mu2 *= -(xi[2]*diameter/xi[3])**3

            mu += mu1 + mu2

        return mu
    
    def chemicalPotential(self):

        mu = np.log(self._density * self.wave**3)
        mu += self.exChemicalPotential()

        return mu

    def exFreeEnergy(self):
        """
        Calculate the excess Helmholtz free energy by using BMCSL equation of state

        input: fluid (a dictionary which contains all property of fluid)
        retruns: the excess Helmholtz free energy density of fluid
        """

        xi = self.xi

        if xi[3] < 1e-3:
            f = 0
        else:
            f = (xi[2]**3/(xi[0]*xi[3]**2) - 1) * np.log(1-xi[3])
            f += 3*xi[1]*xi[2] / (xi[0]*(1-xi[3]))
            f += xi[2]**3 / ( xi[0]*xi[3] * (1-xi[3])**2 )
        
        return f

    def getExChemicalPotential(self, density):

        # print("get chemical potential")
        self.density = density

        return self.exChemicalPotential()

    def getChemicalPotential(self, density):

        self.density = density

        return self.chemicalPotential()

    def getExFreeEnergy(self, density):

        self.density = density

        return self.exFreeEnergy()

if __name__ == "__main__":
    fluid = {}
    fluid["component"] = 2
    fluid["diameter"] = [ 1 ,1 ]
    fluid["wave"] = [ 1 ,1 ]

    eos = BMCSL(fluid)
    # print("Excess chemical potential " + str(eos.getExChemicalPotential([0.1, 0.2])))
    eos.density = [ 0.1 ,0.2 ]

    print("------test pressure------")
    print("compressibility from eos " + str(eos.compressibility()))
    print("compressibility is equal to 1.96671")
    print("------test chemical potential------")
    print("Excess chemical potential from eos " + str(eos.exChemicalPotential()))
    print("Excess chemical potential is equal to 1.74685 and 1.74685")
    print("------test free energy------")
    print("free energy from eos " + str(eos.exFreeEnergy()))
    print("free energy is equal to 0.780134")
