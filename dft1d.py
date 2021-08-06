import numpy as np
import BMCSL_EOS
import MBWR_EOS
import MFA1d
import FMT1d
import FMSA1d
import MSA1d
import Vext
import sys

import time

class dft1d():

    def __init__(self, fluid, system):

        self.learningRate = 0.01
        # self.learningRate = 0.001

        self.fluid = fluid
        self.system = system

        # self.fluid["sigma"] = self.fluid["sigma"].reshape((self.fluid["component"],-1))

        self.bulkDensity = np.array([self.system["bulkDensity"]]).reshape((self.fluid["component"],-1))

        self.density = np.zeros((self.fluid["component"], self.system["grid"]))
        self.oldDensity = np.zeros((self.fluid["component"], self.system["grid"]))

        self.density += self.bulkDensity

        self.error = 1000.0

        if self.fluid["type"] == "LJ":
            Tstar = fluid["temperature"]/ np.array(fluid["epsilon"])
            sigma= np.array(fluid["sigma"])
            self.fluid["diameter"] = (1+0.2977/Tstar) / (1+0.33163/Tstar + 0.0010477/Tstar**2) * sigma

            self.eoshs = BMCSL_EOS.BMCSL(self.fluid)
            if self.system["LJ"] == "FMSA":
                self.lj = FMSA1d.FMSA1d(self.fluid, self.system)
                self.eos = BMCSL_EOS.BMCSL(self.fluid)
            else:
                if self.system["Cor"] == True:
                    self.lj = MFA1d.MFA1d(self.fluid, self.system)
                    self.eos = MBWR_EOS.MBWR_EOS(self.fluid)
                    # self.eos = BMCSL_EOS.BMCSL(self.fluid)
                else:
                    self.lj = MFA1d.MFA1d(self.fluid, self.system)
                    # self.eos = MBWR_EOS.MBWR_EOS(self.fluid)
                    self.eos = BMCSL_EOS.BMCSL(self.fluid)

        elif self.fluid["type"] == "HS":
            self.eos = BMCSL_EOS.BMCSL(self.fluid)
        elif self.fluid["type"] == "ion":
            self.eos = BMCSL_EOS.BMCSL(self.fluid)
            self.ion = MSA1d.MSA1d(self.fluid, self.system)

        self.hs = FMT1d.FMT1d(self.fluid, self.system)

        self.gridWidth = self.system["size"] / self.system["grid"]
        self.maxNum = int(self.system["cutoff"] / self.gridWidth)

        self.exChemicalPotentialBulk = np.zeros((self.fluid["component"], self.system["grid"]))

        exChemP = np.array(self.eos.getExChemicalPotential(self.system["bulkDensity"]) )
        self.exChemicalPotentialBulk += np.array(exChemP.reshape(self.fluid["component"],1))

        # if self.system["LJ"] == "MFA":
            # exChemPlj = (self.fluid["epsilon"] * self.system["bulkDensity"] * self.fluid["sigma"]**3)
            # exChemPlj *= -np.sqrt(2)*(32.0/9.0)*np.pi
            # self.exChemicalPotentialBulk += np.array(exChemPlj)

        ################ initialize the Vext #################

        if self.system["boundaryCondition"] == "bulk":
            self.Vext = np.zeros((self.fluid["component"], self.system["grid"]))
        else:
            vext = Vext.Vext(self.fluid, self.system)
            # self.Vext = np.zeros((self.fluid["component"], self.system["grid"]))
            if self.system["wall"] == "LJ":
                self.Vext = vext.ljWall()
            elif self.system["wall"] == "HS":
                self.Vext = vext.hardWall()
                ############## modified the density distribution #################
                self.density = self.density * np.exp(-self.Vext)
            elif self.system["wall"] == "PW":
                self.Vext = vext.perturbWall(self.system["epsilon"])
            elif self.system["wall"] == "grapheneWall":
                self.Vext = vext.grapheneWall()
            else:
                print("Wrong, system wall")
                sys.exit(-1)

        ################ test Vext #################

        # aa = np.linspace(0, self.system["size"], self.system["grid"])
        # plt.plot(aa, self.Vext[0,:], "r")
        # plt.plot(aa, self.Vext[0,:], "r")
        # plt.plot(aa, self.Vext[0,:], "r")
        # plt.savefig('./Comparison/testRxDDFT/testVext.jpg')

        ############# test Vext, done ##############

    def evolveSystem(self):

        self.oldDensity = self.density.copy()
        self.density = self.bulkDensity * np.exp( - self.Vext \
                                                  - self.exChemicalPotential \
                                                  + self.exChemicalPotentialBulk )

        self.density = self.learningRate * self.density + (1 - self.learningRate) * self.oldDensity

        # if self.system["boundaryCondition"] == "singleWall":
            # self.density[:,-self.maxNum:] = self.system["bulkDensity"].reshape((self.fluid["component"],-1))

        self.error = np.max( np.abs(self.density - self.oldDensity) * (self.fluid["sigma"].reshape(self.fluid["component"],-1))**3 )

    # this function may have problem, need to check it.
    def evolveFsolve(self, density):

        self.density = density.reshape((self.fluid["component"], -1))
        exChemP = self.exChemicalPotential

        # exChemP1 = exChemP + self.Vext
        # return exChemP1.flatten()
        newDensity = self.bulkDensity * np.exp( - self.Vext \
                                                - exChemP \
                                                + self.exChemicalPotentialBulk )

        return ((newDensity - self.density)**2).flatten()


    @property
    def exChemicalPotential(self):

        if self.fluid["type"] == "LJ":
            exChemPhs = self.hs.getExChemicalPotential(self.density)
            exChemPlj = self.lj.getExChemicalPotential(self.density)
            if self.system["LJ"] == "FMSA":
                exChemP = exChemPhs + exChemPlj
            else:
                # exChemP = exChemPhs + exChemPlj
                if self.system["Cor"]:
                    exChemPcor = self.corChemicalPotential
                    exChemP = exChemPhs + exChemPlj + exChemPcor
                else:
                    exChemP = exChemPhs + exChemPlj

        elif self.fluid["type"] == "HS":
            exChemP = self.hs.getExChemicalPotential(self.density)
        elif self.fluid["type"] == "ion":
            exChemPhs  = self.hs.getExChemicalPotential(self.density)
            exChemPIon = self.ion.getExChemicalPotential(self.density)
            exChemP = exChemPhs + exChemPIon

        return exChemP

    '''
    Those two funtions are repeated, one of them should be removed
    '''
    def getExChemP(self, density):

        if self.fluid["type"] == "LJ":
            exChemPhs = self.hs.getExChemicalPotential(density)
            exChemPlj = self.lj.getExChemicalPotential(density)
            if self.system["LJ"] == "FMSA":
                exChemP = exChemPhs + exChemPlj
            else:
                # exChemP = exChemPhs + exChemPlj
                if self.system["Cor"]:
                    exChemPcor = self.corChemicalPotential
                    exChemP = exChemPhs + exChemPlj + exChemPcor
                else:
                    exChemP = exChemPhs + exChemPlj

        elif self.fluid["type"] == "HS":
            exChemP = self.hs.getExChemicalPotential(density)
        elif self.fluid["type"] == "ion":
            exChemPhs  = self.hs.getExChemicalPotential(self.density)
            exChemPIon = self.ion.getExChemicalPotential(self.density) 
            exChemP = exChemPhs + exChemPIon

        return exChemP

    # this function should be checked, it looks like has some error
    @property
    def corChemicalPotential(self):

        rho = self.hs.n3 / (np.pi * (self.fluid["diameter"].reshape((-1,1)))**3/ 6)

        corChemP = np.zeros((self.fluid["component"], self.system["grid"]))

        for i in range(self.system["grid"]):

            epsilon = self.fluid["epsilon"] / self.fluid["temperature"]

            corChemP[:,i]  = self.eos.getExChemicalPotential(rho[:,i])
            corChemP[:,i] -= self.eoshs.getExChemicalPotential(rho[:,i])
            corChemP[:,i] -= -np.sqrt(2)*(32.0/9.0)*np.pi*epsilon * rho[:,i] * self.fluid["sigma"]**3

        densityMatrix = self.hs.densityIntegrate(corChemP, self.fluid["component"], self.hs.maxNum, self.system["grid"])

        n3 = np.pi * np.sum(densityMatrix * self.hs.n3parameter, axis = 0)

        return n3

    def printFinalResult(self):
        pass


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    def testDFT(fluid, system):

        test = dft1d(fluid, system)

        # test.evolveSystem()
        # test.evolveSystem()

        while test.error > 10e-5:
            test.evolveSystem()
            # print("system evolved, the error is ")
            # print(test.error)

        return test.density

    fluid = {}
    fluid["component"] = 1
    fluid["type"] = "LJ"
    # fluid["type"] = "HS"
    fluid["sigma"] = np.array([1.0])
    fluid["diameter"] = np.array([1.0])
    fluid["wave"] = [1.0]
    fluid["temperature"] = 1.2
    fluid["epsilon"] = np.array([1.0])

    system = {}
    system["bulkDensity"] = np.array([0.5925])
    system["boundaryCondition"] = "slitPore"
    system["wall"] = "LJ"

    def plotResult(fluid, system, label):

        if label:
            s = ''
            file = "./Comparison/LJ_cDFT/slitporeLJ" + str(system["size"]) + ".dat"
        else:
            s = '_SingleWall'
            file = "./Comparison/LJ_cDFT/test_LJ_singleWall_30.dat"

        a = np.linspace(0, system["size"], system["grid"])
        c = np.loadtxt(file)
        t = testDFT(fluid, system)

        plt.figure()
        plt.xlim((0,system["size"]))
        plt.ylim((0,4.5))
        plt.plot(c[:,0], c[:,1], linestyle="None", marker="o")
        plt.plot(a, t.T)
        if system["LJ"] == "FMSA":
            file1 = './Comparison/LJ_cDFT/testLJ' + str(system["size"]) + s + '_FMSA.jpg'
        else:
            if system["Cor"]:
                file1 = './Comparison/LJ_cDFT/testLJ' + str(system["size"]) + s + '_MFA_Cor.jpg'
            else:
                file1 = './Comparison/LJ_cDFT/testLJ' + str(system["size"]) + s + '_MFA.jpg'
        plt.savefig(file1)

    def testDifferentMethod(fluid, system, label):
        system["LJ"] = "MFA"

        system["Cor"] = True
        plotResult(fluid, system, label)

        system["Cor"] = False
        plotResult(fluid, system, label)

        system["LJ"] = "FMSA"
        plotResult(fluid, system, label)


    # ==================== test slit pore system ==================== #

    '''
    The simulation data can be found in Y. Tang et al. PRE, 70, 011201
    (2004) Figure 6 and Figure 7.
    '''

    # start = time.time()

    # --------------------   test H = 7.5 sigma  -------------------- #
    # system["grid"] = 750
    # system["cutoff"] = np.array([5.0])



    # system["size"] = 7.5
    # system["grid"] = 375
    # system["cutoff"] = np.array([3.0])

    # testDifferentMethod(fluid, system, 1)

    # --------------------   test H = 4.0 sigma  -------------------- #
    system["size"] = 4.0
    system["grid"] = 200
    system["cutoff"] = np.array([3.0])

    testDifferentMethod(fluid, system, 1)

    # ================= test slit pore system, done ================= #

    # print(time.time() - start)

    # ========================= test single wall ===================== #

    # ============ generate the test density profile ============ #
    # system["grid"] = 3000
    # system["boundaryCondition"] = "slitPore"
    # system["size"] = 30
    # system["cutoff"] = np.array([5.0])

    # system["LJ"] = "FMSA"

    # test = dft1d(fluid, system)
    # test.evolveSystem()

    # while test.error > 10e-5:
        # test.evolveSystem()

    # a = np.linspace(0, system["size"], system["grid"]).reshape((-1,1))
    # t = test.density
    # b = np.hstack((a, t.T))
    # np.savetxt('./Comparison/LJ_cDFT/test_LJ_singleWall_30.dat', b)
    # ========== generate the test density profile, end ========== #

    # start = time.time()

    # system["grid"] = 750

    # system["boundaryCondition"] = "singleWall"
    # system["size"] = 15
    # system["cutoff"] = np.array([5.0])
    # system["wall"] = "LJ"

    # testDifferentMethod(fluid, system, 0)

    # print(time.time() - start)
