import numpy as np
from scipy.integrate import quad
from scipy.integrate import simpson
from scipy.integrate import trapezoid
import sys
import Functional
from scipy.optimize import fsolve

class MSA1d(Functional.Functional):

    def __init__(self, fluid, system):

        super(MSA1d, self).__init__(fluid, system)

        # ============ init DCF ============ #

        self.DCF = np.zeros((self.maxNum*2+1, self.fluid["component"], self.fluid["component"]))

        self.unitLength = 5 * 10**(-10)

        self.lb  = (1.60217653*10**(-19))**2 * 6.022140*10**23 / 1000
        self.lb /= ( 4*np.pi * (8.854187817*10**(-12)) * self.fluid["dielec"] * self.fluid["temperature"] * self.unitLength )
        # Currently, fluid["dielec"] must be a number 

        self.X   = fsolve(self.solveX, np.zeros(self.fluid["component"]))

        self.N   = (self.X - self.fluid["charge"]) / self.fluid["diameter"]

        self.Gamma = (np.pi * self.lb * np.sum(self.system["bulkDensity"] * self.X**2))**(1/2)

        self.L   = np.zeros((4, self.fluid["component"], self.fluid["component"]))

        self.S   = self.N + self.Gamma * self.X

        self.vl = self.system["WallVolt"][0] * (1.60217652*10**(-19) * 6.02*10**23 / 1000 ) / (self.fluid["temperature"])
        self.vr = self.system["WallVolt"][1] * (1.60217652*10**(-19) * 6.02*10**23 / 1000 ) / (self.fluid["temperature"])

        '''
        if x = [x0, x1]
          xi - xj = [[x0-x0, x0-x1]
                     [x1-x0, x1-x1]]
        Hence, xi = [[x0 x0]
                     [x1 x1]]

               xj = [[x0 x1
                     [x0 x1]]
        '''

        # vector version:
        Xi = self.X.reshape([-1,1])
        Xi = np.repeat(Xi, [self.fluid["component"]], axis=1)
        Xj = Xi.T

        di = self.fluid["diameter"].reshape([-1, 1])
        di = np.repeat(di, [self.fluid["component"]], axis=1)
        dj = di.T

        Si = self.S.reshape([-1,1])
        Si = np.repeat(Si, [self.fluid["component"]], axis=1)
        Sj = Si.T

        Ni = self.N.reshape([-1,1])
        Ni = np.repeat(Ni, [self.fluid["component"]], axis=1)
        Nj = Ni.T

        self.L[0,:,:] = ( (Xi+Xj)/4 )*(Si-Sj) - ( (di-dj)/16 )*( (Si+Sj)**2 - 4*Ni*Nj )
        self.L[1,:,:] = (Xi-Xj)*(Ni-Nj) + (Xi**2+Xj**2)*self.Gamma + (di+dj)*Ni*Nj - (di*Si**2 + dj*Sj**2)/3
        '''
        Here, it should be noted that the sign of the last term in L[1] is negtive. 
        However, in many papers of Prof. Jiang and Prof. Jin's thesis, it is postive.
        '''
        self.L[2,:,:] = (Xi/di)*Si + (Xj/dj)*Sj + Ni*Nj - (Si**2 + Sj**2)/2
        self.L[3,:,:] = Si**2 / (6*di**2) + Sj**2 / (6*dj**2)

        for i in range(self.fluid["component"]):
            for j in range(self.fluid["component"]):

                u = [quad(self.cmsaz,0,np.inf,args=(np.abs(z)*self.gridWidth, i, j))[0] \
                     for z in range(-self.maxNum, self.maxNum+1)]

                u = - np.array(u)

                u *= 2 * np.pi
                self.DCF[:,i,j] = u

        self.DCF[0,:,:] = self.DCF[0,:,:]/2
        self.DCF[-1,:,:] = self.DCF[-1,:,:]/2


    def solveX(self, X):
        Gamma = (np.pi * self.lb * np.sum(self.system["bulkDensity"] * X**2))**(1/2)
        alpha = (np.pi/2) * (1 - (np.pi/6)*np.sum(self.system["bulkDensity"]*self.fluid["diameter"]**3) )**(-1)
        return (1+Gamma*self.fluid["diameter"])*X + alpha*self.fluid["diameter"]**2 * np.sum(self.system["bulkDensity"]*self.fluid["diameter"]*X) - self.fluid["charge"]


    def cmsa(self, r ,i, j):

        r = 1e-5 if r < 1e-5 else r

        di = self.fluid["diameter"][i]
        dj = self.fluid["diameter"][j]

        if r < np.abs(di-dj)/2:
            c = -self.fluid["charge"][i]*self.N[j] + self.X[i]*(self.N[i] + self.Gamma*self.X[i])
            c = c - (di/3) * (self.N[i] + self.Gamma*self.X[i])**2
            c = -2 * self.lb * c + self.lb * self.fluid["charge"][i]*self.fluid["charge"][j] / r
        elif r < (di+dj)/2:
            c = (di-dj)*self.L[0,i,j] - r*self.L[1,i,j] + (r**2) * self.L[2,i,j] + (r**4)*self.L[3,i,j]
            c = self.lb * c / r + self.lb * self.fluid["charge"][i]*self.fluid["charge"][j] / r
        else:
            c = 0

        return c


    def cmsaz(self, rr, z, i, j):

        r = np.abs( np.sqrt(rr**2 + z**2) )
        u = self.cmsa(r, i, j)

        return u * rr

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, density):
        self._density = density

    '''
    old version
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

    def exChemicalPotential(self):

        density = self._density - self.system["bulkDensity"].reshape((self.fluid["component"], -1))
        density[:,:26] = 0
        density[:,226:] = 0
        return self.densityIntegrateFFT(density, self.DCF, self.fluid["component"], self.maxNum, self.system["grid"])

    @property
    def solvePoisson(self):

        coe      = (1.60217653*10**(-19))**2 * (6.02 * 10**23 / 1000) / ( 8.854*10**(-12)*self.fluid["temperature"] * self.unitLength )

        # density  = np.loadtxt("./Comparison/ion/potential_profile_PB.dat")
        # density  = density[:,1]
        density  = np.sum(self._density * self.fluid["charge"].reshape((-1,1)), axis = 0)
        density *= coe

        #################### Riemann sum ####################
        # z     = np.linspace(0, self.system["size"], self.system["grid"])
        # reRho = trapezoid(density * (self.system["size"]-z) )
        # phiC  = - np.sum(reRho) * self.gridWidth

        # phiC  = phiC + self.vl - self.vr
        # phiC /= (self.system["size"])

        # phi  = -np.array([ (trapezoid( density[:i] * (z[i]-z[:i]), z[:i] ) if i > 0 else 0) for i in range(self.system["grid"])])
        # phi  = phi + self.vl - phiC * z
        ################# Riemann sum, done #################

        #################### Two-points Gaussian Integrate ####################
        a = (1/2 - np.sqrt(3)/6)
        b = (1/2 + np.sqrt(3)/6)

        ldensity = np.hstack(( np.zeros(1), density[:-1] ))

        rhoR1 = a * density + (1 - a) * ldensity
        rhoR2 = b * density + (1 - b) * ldensity

        z  = np.linspace(-1, self.system["grid"]-1, self.system["grid"])
        p1 = (z + a) * self.gridWidth
        p2 = (z + b) * self.gridWidth

        rhostar    = ( rhoR1 * (self.system["size"] - p1) + rhoR2 * (self.system["size"] - p2) ) * self.gridWidth / 2
        rhostar[0] = 0

        phiC  = - np.sum( rhostar ) + self.vl - self.vr
        phiC /= self.system["size"]

        z1     = np.linspace(0, self.system["size"], self.system["grid"])
        phi    = np.array([ ( np.sum( rhoR1[:i]*(z1[i]-p1[:i]) + rhoR2[:i]*(z1[i]-p2[:i]) ) if i > 0 else 0) for i in range(self.system["grid"]) ])
        phi   *= -1 * self.gridWidth / 2
        phi[0] = 0
        phi    = phi + self.vl - phiC * z1
        ################# Two-points Gaussian Integrate, done #################

        return phi

    def getExChemicalPotential(self, density):

        self.density = density
        exChemP = self.exChemicalPotential()
        phi = self.solvePoisson

        phi1 = np.zeros((self.fluid["component"], self.system["grid"])) + phi
        phi1 *= self.fluid["charge"].reshape((self.fluid["component"], -1))

        return exChemP + phi1

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import MBWR_EOS
    import FMT1d

    fluid = {}
    fluid["component"] = 2
    fluid["type"] = "ion"
    fluid["sigma"] = np.array([1.0, 1.0])
    fluid["diameter"] = np.array([1.0, 1.0])
    fluid["wave"] = [1.0, 1.0]
    fluid["temperature"] = (298/120.272239)

    fluid["charge"] = np.array([1.0, -1.0])
    fluid["dielec"] = 1

    system = {}
    system["grid"] = 251
    system["bulkDensity"] = np.array([0.29, 0.29])

    system["boundaryCondition"] = "slitPore"
    system["size"] = 5.0
    system["cutoff"] = np.array([1.3])
    system["wall"] = "HS"

    system["WallVolt"] = [-0.3, 0.3]

    test = MSA1d(fluid, system)

    # ==================== test c_r ==================== #
    # c = np.loadtxt("./Comparison/ion/cr2.dat")

    # b1 = np.linspace(0, 1.5, 1000)
    # b2 = np.zeros(len(b1))
    # for i in range(len(b1)):
        # b2[i] = test.cmsa(b1[i], 0, 1)

    # plt.figure()
    # plt.plot(b1, b2)
    # plt.plot(c[:80, 0], c[:80, 1], 'o')
    # plt.xlim((0, 1.5))
    # plt.ylim((-30, 2))
    # plt.savefig("./Comparison/ion/testCr.jpg")
    # ================= test c_r, done ================= #


    # ==================== test PB equation ==================== #
    # ------------------ test the init state ------------------ #
    # d  = np.loadtxt("./Comparison/ion/potential_profile_init.dat")
    # d2 = test.solvePoisson
    # d3 = np.linspace(0, system["size"], len(d2))

    # plt.figure()
    # plt.plot(d[:,0], d[:,1], "o")
    # plt.plot(d3,d2)
    # plt.savefig("./Comparison/ion/testPS_init.jpg")
    # --------------- test the init state, done --------------- #

    # ------------------ test the final state ------------------ #
    # f = np.loadtxt("./Comparison/ion/Final_equilibrium.dat")
    # density = np.zeros((fluid["component"], system["grid"]))
    # density[:, 25:226] = f[:, 1:3].T

    # test.density = density
    # d2 = test.solvePoisson
    # d3 = np.linspace(0, system["size"], len(d2))

    # plt.figure()
    # plt.plot(d3,d2)
    # plt.plot(f[:,0]+0.5, f[:,3], "r")
    # plt.savefig("./Comparison/ion/testPS_Fin_test.jpg")
    # --------------- test the final state, done --------------- #
    # ================= test PB equation, done ================= #


    # ==================== test the chemical potential ==================== #
    f = np.loadtxt("./Comparison/ion/Final_equilibrium.dat")
    f1 = np.loadtxt("./Comparison/ion/chemPMSA_J.dat")
    density = np.zeros((fluid["component"], system["grid"]))
    density[:, 25:226] = f[:, 1:3].T

    test.density = density
    d2 = test.exChemicalPotential()
    d3 = np.linspace(0, system["size"], system["grid"])

    plt.figure()
    plt.plot(f1[:,0]+0.5, f1[:,1], color = "b", linestyle = ":")
    plt.plot(d3,d2[0,:], 'b')

    plt.plot(f1[:,0]+0.5, f1[:,2], color = "r", linestyle = ":")
    plt.plot(d3,d2[1,:], "r")
    plt.savefig("./Comparison/ion/testChemPMSA.jpg")
    # ================= test the chemical potential,done ================= #

    # ==================== test the chemical potential ==================== #
    f = np.loadtxt("./Comparison/ion/Final_equilibrium.dat")
    density = np.zeros((fluid["component"], system["grid"]))
    density[:, 25:226] = f[:, 1:3].T

    test.density = density
    d2 = test.getExChemicalPotential(density)
    d3 = np.linspace(0, system["size"], system["grid"])

    plt.figure()
    plt.plot(d3[25:226],d2[0,25:226])

    plt.plot(d3[25:226],d2[1,25:226], "r")
    plt.savefig("./Comparison/ion/testChemPE.jpg")
    # ================= test the chemical potential,done ================= #
