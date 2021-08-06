import dft1d
import numpy as np
import BMCSL_EOS
import MBWR_EOS
import FMT1d
import FMSA1d
import Vext
import sys
import ADAM

class DDFT(dft1d.dft1d):

    """
    This class is used to describe the dynamics adsorption model
    """

    def __init__(self, fluid, system):

        dft1d.dft1d.__init__(self, fluid, system)

        # vext = Vext.Vext(self.fluid, self.system)
        # self.Vext = vext.perturbwall(-2)

        # self.density = self.system["initdensity"] > 10e-3
        self.density = self.system["initdensity"]

        self.k = 0
        self.deltaT = self.system["deltaT"]

        self.drive = np.zeros((self.fluid["component"], self.system["grid"]))
        # self.wave = self.fluid["wave"]

        self.solve = ADAM.adam(self.drive, self.deltaT)

        self.D = self.system["mobility"]

        self.totalDensity = 0

        self.upperBound = 10 * self.fluid["temperature"]

        self.chemicalPotentialBulk = self.eos.getChemicalPotential(self.system["bulkDensity"]);

        # self.Vext[self.Vext > 10] = 10

    def getChemicalPotential(self, density):

        exChemP = self.getExChemP(density)
        chemP = np.zeros((self.fluid["component"], self.system["grid"]))

        label = self.Vext < self.upperBound

        rd = density * (self.fluid["wave"].reshape((self.fluid["component"],-1)))**3

        label1 = rd < 0.001

        chemP[label] = np.log(rd[label])

        chemP[label] += exChemP[label]
        chemP[label] += self.Vext[label]

        chemP[self.Vext > self.upperBound] = 20

        return chemP

    def calculateDerivative(self, density, component):

        rho = density
        rhol = np.hstack(( np.zeros((component, 1)), density[:, :-1] ))
        rhor = np.hstack(( density[:,1:], np.zeros((component, 1)) ))

        drho = (rhor - rhol) / (2 * self.gridWidth)

        return drho

    def evolveDrive(self, density):

        density[self.Vext > self.upperBound] = 0
        chemP = self.getChemicalPotential(density)

        # =================== As Jiang Jian's code ============================#
        lchemP = np.hstack(( chemP[:, 1:], np.zeros((self.fluid["component"], 1)) ))
        ldensity = np.hstack(( density[:, 1:], np.zeros((self.fluid["component"], 1))  ))

        if self.system["boundaryCondition"] == "singleWall":
            lchemP[:,-1] = self.chemicalPotentialBulk

        flux = ( lchemP - chemP ) / self.gridWidth
        flux = flux * ( ldensity + density ) / 2.0
        flux[self.Vext > self.upperBound] = 0

        if self.system["wall"] == "LJ":
            for i in range(self.fluid["component"]):
                a = np.where(self.Vext[i,:] == ( (self.Vext[i,:])[self.Vext[i,:] < self.upperBound][0] ))
                b = np.where(self.Vext[i,:] == ( (self.Vext[i,:])[self.Vext[i,:] < self.upperBound][-1] ))

                if self.system["boundaryCondition"] == "singleWall":
                    flux[i,a] = 0
                else:
                    flux[i,a] = 0
                    flux[i,b[0]-1] = 0
                    flux[i,b] = 0
        elif self.system["wall"] == "HS":
            for i in range(self.fluid["component"]):
                grid = round(self.fluid["diameter"][i]) / (2*self.gridWidth)
                grid = int(grid)

                if self.system["boundaryCondition"] == "singleWall":
                    flux[i, :grid+1] = 0
                    # flux[i, -1]   = 0
                else:
                    flux[i, :grid+1]      = 0
                    flux[i, -(grid+1)] = 0
                    flux[i, -(grid)]   = 0

        rflux = np.hstack(( np.zeros((self.fluid["component"], 1)), flux[:, :-1] ))

        drive = (flux - rflux) / self.gridWidth
        # =====================================================================#

        drive *= self.D

        return drive

    def evolveDriveIvp(self, t, density):

        density1 = density.reshape((self.fluid["component"], -1))

        drive = self.evolveDrive(density1)

        drive1 = drive.flatten()

        return drive1

    def evolveSystem(self):

        drive = self.evolveDrive(self.density)

        newDensity = self.solve.adamsBash(self.density, drive, self.k)
        # newDensity[self.Vext>1] = self.system["initdensity"][self.Vext>1]
        for j in range(3):
            oldDensity = newDensity.copy()
            newDerive = self.evolveDrive(newDensity)
            newDensity = self.solve.adamsMoulton(self.density, drive, newDerive, self.k)

            # newDensity[self.Vext>20] = self.system["initdensity"][self.Vext>20]
            # testValue = np.abs(np.sum(oldDensity) - np.sum(newDensity))/np.sum(oldDensity)
            # if testValue < 0.05:
                # break

        newDerive = self.evolveDrive(newDensity)
        self.solve.updateDerive(newDerive)
        self.density = newDensity.copy()
        self.totalDensity = np.sum(self.density) * self.gridWidth

        self.k = self.k + 1


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # fluid = {}
    # fluid["component"] = 2
    # fluid["type"] = "HS"
    # fluid["sigma"] = np.array([1.0, 2.0])
    # fluid["diameter"] = np.array([1.0, 2.0])
    # fluid["wave"] = np.array([1.0, 2.0])
    # # fluid["wave"] = 1.0
    # fluid["temperature"] = 1.0
    # # fluid["lambda"] = np.array([1.0])

    # fluid["mass"] = np.array([1.0, 1.0])

    # system = {}
    # system["grid"] = 2048
    # # system["bulkDensity"] = np.array([0.1, 0.1])
    # system["bulkDensity"] = (np.array([0.042, 0.026])) / (fluid["diameter"]/2)**3
    # """
    # This bulk density is used to reproduce the Fig.6(a) of R.Roth PRE 80, 021409 (2009)
    # """
    # # print(system["bulkDensity"])

    # system["boundaryCondition"] = "slitPore"
    # system["size"] = 20.48
    # system["cutoff"] = np.array([3.0])

    # system["wall"] = "PW"
    # # system["epsilon"] = -2
    # system["epsilon"] = 0
    # system["deltaT"] = 10e-6
    # # system["deltaT"] = 10e-5
    # system["mobility"] = 1

    # test1 = dft1d.dft1d(fluid, system)

    ##===================Test Vext=====================##
    # a = np.linspace(0, system["size"], system["grid"])

    # y1 = np.loadtxt('./Comparison/testDDFT/Vext_test.dat')
    # y2 = np.loadtxt('./Comparison/testDDFT/Vext_test2.dat')

    # plt.figure()
    # plt.xlim((2,38))
    # plt.ylim((-2,3))
    # plt.plot(a*2, test1.Vext[0,:], color='r')
    # plt.plot(a*2, test1.Vext[1,:], color='k', linestyle='dashed')
    # plt.plot(y1[:,0], y1[:,1], 'o')
    # plt.plot(y2[:,0], y2[:,1], 's')
    # plt.savefig('./Comparison/testDDFT/vextTestS1.jpg')
    ##=================================================##

    # while test1.error > 10e-5:
        # test1.evolveSystem()

    ##===================Check equilibrium density profile=====================##
    # a = np.linspace(0, system["size"], system["grid"])
    # t = test1.density
    # yB = np.loadtxt('./Comparison/testDDFT/densityStartBig.dat')
    # yS = np.loadtxt('./Comparison/testDDFT/densityStartSmall.dat')

    # plt.figure()
    # plt.plot(a*2, t[0,:]*(fluid["diameter"][0]/2)**3, color='r')
    # plt.plot(a*2, t[1,:]*(fluid["diameter"][1]/2)**3, color='k', linestyle='dashed')
    # plt.plot(yB[:,0], yB[:,1], color='k', linestyle=":")
    # plt.plot(yS[:,0], yS[:,1], color='r', linestyle="-.")
    # plt.savefig('./Comparison/testDDFT/eq_densityStart.jpg')
    ##========================================================================##

    # ==================== test adam ==================== #
    # np.save("./testinitdensity.npy", test1.density)
    # td = np.load("./testinitdensity.npy")

    # system["epsilon"] = -2
    # system["initdensity"] = td

    # test2 = DDFT(fluid, system)

    # print("system on")
    # totTime = int(0.5 / system["deltaT"])
    # totTime = int(2.0 / system["deltaT"])
    # totTime = int(8.0 / system["deltaT"])
    # totTime = 5000
    # totTime = 7000
    # totTime = 70

    # testdensity = np.zeros((2, totTime))

    # for i in range(totTime):
        # test2.evolveSystem()
        # if i%1001 == 0:
            # print("run")
            # print(i * system["deltaT"])

        # testdensity[1,i] = test2.totalDensity
        # testdensity[0,i] = i
        # # print(test2.totalDensity)

    # np.save("densityFin.npy", test2.density)
    # np.save("totalDensityFin.npy", testdensity)
    # np.save("testVext.npy", test2.Vext)

    # a = np.linspace(0, system["size"], system["grid"])
    # t = test2.density
    # # yB = np.loadtxt('./Comparison/testDDFT/density2Big.dat')
    # # yS = np.loadtxt('./Comparison/testDDFT/density2Small.dat')
    # # yB = np.loadtxt('./Comparison/testDDFT/density10Big.dat')
    # # yS = np.loadtxt('./Comparison/testDDFT/density10Small.dat')
    # yB = np.loadtxt('./Comparison/testDDFT/densityFinBig.dat')
    # yS = np.loadtxt('./Comparison/testDDFT/densityFinSmall.dat')

    # plt.figure()
    # plt.plot(a*2, t[0,:]*(fluid["diameter"][0]/2)**3, color='r')
    # plt.plot(a*2, t[1,:]*(fluid["diameter"][1]/2)**3, color='k')
    # plt.plot(yB[:,0], yB[:,1], color='k', linestyle=":")
    # plt.plot(yS[:,0], yS[:,1], color='r', linestyle="-.")
    # plt.savefig('./Comparison/testDDFT/densityFin.jpg')
    # # plt.savefig('./Comparison/testDDFT/density100.jpg')

    # plt.figure()
    # plt.plot(testdensity[0,:], testdensity[1,:])
    # plt.savefig("./Comparison/testDDFT/testTotalDensity.jpg")
    # ================== test adam done ================== #


    # ==================== test ivp ==================== #

    # from scipy.integrate import solve_ivp

    # td1 = td.flatten()

    # sol = solve_ivp(test2.evolveDriveIvp, [0, 2.5], td1)

    # t = sol.y[:,-1].reshape((fluid["component"], -1))

    # a = np.linspace(0, system["size"], system["grid"])
    # # yB = np.loadtxt('./Comparison/testDDFT/density2Big.dat')
    # # yS = np.loadtxt('./Comparison/testDDFT/density2Small.dat')
    # yB = np.loadtxt('./Comparison/testDDFT/density10Big.dat')
    # yS = np.loadtxt('./Comparison/testDDFT/density10Small.dat')

    # plt.figure()
    # plt.plot(a*2, t[0,:]*(fluid["diameter"][0]/2)**3, color='r')
    # plt.plot(a*2, t[1,:]*(fluid["diameter"][1]/2)**3, color='k')
    # plt.plot(yB[:,0], yB[:,1], color='k', linestyle=":")
    # plt.plot(yS[:,0], yS[:,1], color='r', linestyle="-.")
    # plt.savefig('./Comparison/testDDFT/densityTestIvp.jpg')

    # ==================== test end ==================== #

    # ==================== test  LJ ==================== #

    # fluid = {}
    # fluid["component"]   = 1
    # fluid["type"]        = "LJ"
    # fluid["sigma"]       = np.array([1.0])
    # fluid["diameter"]    = np.array([1.0])
    # fluid["wave"]        = np.array([1.0])
    # fluid["temperature"] = 1.2
    # fluid["epsilon"]     = np.array([1.0])


    # system = {}
    # system["grid"]              = 750
    # system["bulkDensity"]       = np.array([0.5925])
    # system["boundaryCondition"] = "slitPore"
    # system["size"]              = 7.5
    # system["cutoff"]            = np.array([5.0])
    # system["wall"]              = "LJ"
    # system["LJ"]                = "MFA"
    # system["deltaT"]            = 10e-6
    # system["mobility"]          = 1
    # system["Cor"]               = True


    # # -------- calculate initdensity -------- #
    # # test1 = dft1d.dft1d(fluid, system)
    # # while test1.error > 10e-6:
        # # test1.evolveSystem()

    # # np.save("./Comparison/testDDFT/LJ/initdensity.npy", test1.density)
    # # -------- calculate initdensity -------- #

    # # -------- set initdensity -------- #
    # t = np.load("./Comparison/testDDFT/LJ/initdensity.npy")
    # system["initdensity"]  = np.zeros_like(t);
    # system["initdensity"] += ([np.sum(t) / 650]);

    # # t = np.load("./Comparison/testDDFT/LJ/testdensity.npy")
    # # -------------- done ------------- #

    # # ------ test equilibrium density deistribution ------ #
    # c = np.loadtxt("./Comparison/slitporeLJ7.0.dat")
    # # t = test1.density
    # # a = np.linspace(0, system["size"], system["grid"])

    # # plt.figure()
    # # plt.xlim((0,7.5))
    # # plt.ylim((0,4))
    # # plt.plot(c[:,0], c[:,1], linestyle="None", marker="o")
    # # plt.plot(a, t.T)
    # # # plt.plot(a, system["initdensity"].T)
    # # plt.savefig('./Comparison/testDDFT/LJ/testDensityEq7.jpg')
    # # ------------------- test finished ------------------ #

    # from scipy.integrate import solve_ivp

    # test2 = DDFT(fluid, system)
    # # np.save("./Comparison/testDDFT/LJ/testVest.npy", test2.Vext)

    # td1 = system["initdensity"].flatten()
    # sol = solve_ivp(test2.evolveDriveIvp, [0, 0.5], td1)
    # t = sol.y[:,-1].reshape((fluid["component"], -1))

    # np.save("./Comparison/testDDFT/LJ/testdensity.npy", t)

    # a = np.linspace(0, system["size"], system["grid"])
    # plt.figure()
    # plt.xlim((0,7.5))
    # plt.ylim((0,4))
    # plt.plot(a, t.T, color='r')
    # plt.plot(c[:,0], c[:,1], linestyle="None", marker="o")
    # plt.savefig('./Comparison/testDDFT/LJ/testDensityT.jpg')

    # ==================== test end ==================== #


    # ==================== test single Wall (HS) ==================== #

    # fluid = {}
    # fluid["component"]   = 1
    # fluid["type"]        = "HS"
    # fluid["sigma"]       = np.array([1.0])
    # fluid["diameter"]    = np.array([1.0])
    # fluid["wave"]        = np.array([1.0])
    # fluid["temperature"] = 1.0
    # fluid["epsilon"]     = np.array([1.0])

    # system = {}
    # system["grid"]              = 450
    # system["bulkDensity"]       = np.array([0.30])
    # system["boundaryCondition"] = "singleWall"
    # system["size"]              = 4.5
    # system["cutoff"]            = np.array([5.0])
    # system["wall"]              = "HS"
    # system["deltaT"]            = 10e-6
    # system["mobility"]          = 1

    # # -------- calculate initdensity -------- #
    # # test1 = dft1d.dft1d(fluid, system)
    # # while test1.error > 10e-6:
        # # test1.evolveSystem()

    # # np.save("./Comparison/testDDFT/SingleWallHS/initdensity.npy", test1.density)

    # # a = np.linspace(0, system["size"], system["grid"])
    # # plt.figure()
    # # plt.figure()
    # # plt.plot(a, test1.density[0,:])
    # # plt.savefig('./Comparison/testDDFT/SingleWallHS/testInitDensity.jpg')

    # # -------- calculate initdensity -------- #

    # # -------- set initdensity -------- #
    # t1 = np.load("./Comparison/testDDFT/SingleWallHS/initdensity.npy")
    # system["initdensity"]  = np.zeros_like(t1);
    # # system["initdensity"] += ([np.sum(t1) / 400]);
    # system["initdensity"] += ([0.30]);
    # # -------------- done ------------- #

    # from scipy.integrate import solve_ivp

    # test2 = DDFT(fluid, system)
    # # np.save("./Comparison/testDDFT/LJ/testVest.npy", test2.Vext)

    # td1 = system["initdensity"].flatten()
    # sol = solve_ivp(test2.evolveDriveIvp, [0, 6.0], td1)
    # t = sol.y[:,-1].reshape((fluid["component"], -1))

    # np.save("./Comparison/testDDFT/SingleWallHS/testdensity.npy", t)

    # a = np.linspace(0, system["size"], system["grid"])
    # plt.figure()
    # # plt.xlim((0,4.5))
    # # plt.ylim((0,4))
    # plt.plot(a, t1[0,:], linestyle="None", marker="o")
    # # plt.plot(a, t1[0,:], linestyle="None", marker="o")
    # plt.plot(a, t.T, color='r')
    # plt.savefig('./Comparison/testDDFT/SingleWallHS/testDensityT.jpg')

    # ========================== test end =========================== #

    # ==================== test single Wall (LJ) ==================== #

    fluid = {}
    fluid["component"]   = 1
    fluid["type"]        = "LJ"
    fluid["sigma"]       = np.array([1.0])
    fluid["diameter"]    = np.array([1.0])
    fluid["wave"]        = np.array([1.0])
    fluid["temperature"] = 1.2
    fluid["epsilon"]     = np.array([1.0])

    system = {}

    # system["grid"]              = 750
    system["grid"]              = 375
    system["size"]              = 7.5

    system["bulkDensity"]       = np.array([0.5925])
    system["boundaryCondition"] = "singleWall"
    system["cutoff"]            = np.array([3.0])
    system["wall"]              = "LJ"
    system["LJ"]                = "MFA"
    system["Cor"]               = False
    # system["LJ"]                = "FMSA"
    system["deltaT"]            = 10e-6
    system["mobility"]          = 1

    # -------- calculate initdensity -------- #
    # test1 = dft1d.dft1d(fluid, system)
    # while test1.error > 10e-6:
        # test1.evolveSystem()

    # np.save("./Comparison/testDDFT/SingleWallLJ/initdensity.npy", test1.density)

    # a = np.linspace(0, system["size"], system["grid"])
    # plt.figure()
    # plt.figure()
    # plt.plot(a, test1.density[0,:])
    # plt.savefig('./Comparison/testDDFT/SingleWallLJ/testInitDensity.jpg')
    # -------- calculate initdensity -------- #

    # -------- set initdensity -------- #
    t1 = np.load("./Comparison/testDDFT/SingleWallLJ/initdensity.npy")
    # system["initdensity"]  = np.zeros_like(t1);
    # # system["initdensity"] += ([np.sum(t1) / 400]);

    system["initdensity"]  = np.zeros((fluid["component"], system["grid"]))
    system["initdensity"] += np.array([0.5925]);

    # tt = np.load("./Comparison/testDDFT/SingleWallLJ/testdensity.npy")
    # system["initdensity"] = tt
    # # -------------- done ------------- #

    from scipy.integrate import solve_ivp

    test2 = DDFT(fluid, system)
    # np.save("./Comparison/testDDFT/LJ/testVest.npy", test2.Vext)

    td1 = system["initdensity"].flatten()
    # sol = solve_ivp(test2.evolveDriveIvp, [0, 1.5], td1)
    sol = solve_ivp(test2.evolveDriveIvp, [0, 10], td1)
    t = sol.y[:,-1].reshape((fluid["component"], -1))

    np.save("./Comparison/testDDFT/SingleWallLJ/testdensity.npy", t)

    a = np.linspace(0, system["size"], system["grid"])
    a1 = np.linspace(0, system["size"], 750)
    plt.figure()
    plt.plot(a1, t1[0,:], linestyle="None", marker="o")
    plt.plot(a, t.T, color='r')
    plt.savefig('./Comparison/testDDFT/SingleWallLJ/testDensityT.jpg')

    # ========================== test end =========================== #

