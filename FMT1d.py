
import numpy as np
import sys
import Functional

class FMT1d(Functional.Functional):

    def __init__(self, fluid, system):

        super(FMT1d, self).__init__(fluid, system)

        ################ initialize the FMT parameter #################

        num = (self.fluid["diameter"]/2.0) / self.gridWidth
        self.num = num.astype(np.int)
        # print(self.num.dtype)

        self.n2parameter = np.zeros((np.max(self.num)*2+1, self.fluid["component"], 1))
        self.n3parameter = np.zeros((np.max(self.num)*2+1, self.fluid["component"], 1))
        self.nv2parameter = np.zeros((np.max(self.num)*2+1, self.fluid["component"], 1))

        self.maxNum = int(round( np.max(self.num) ))
        self.n = np.zeros((6, self.system["grid"]))
        for i in range(self.fluid["component"]):
            num = int(round(self.num[i]))
            radius2 = (self.fluid["diameter"][i]/(2*self.gridWidth)) ** 2

            self.n2parameter[self.maxNum-num:self.maxNum+num +1, i] = 1 * self.gridWidth
            self.nv2parameter[self.maxNum-num:self.maxNum+num +1, i] = np.array([[x for x in \
                                                                 range(-num, num+1)]]).T * self.gridWidth**2
            self.n3parameter[self.maxNum-num:self.maxNum+num +1, i] = np.array([[(radius2 -x**2) for x in \
                                                                range(-num, num+1)]]).T * self.gridWidth**3

            # boundary condition
            self.n2parameter[[self.maxNum-num, self.maxNum+num], i] = \
                self.n2parameter[[self.maxNum-num, self.maxNum+num], i] /2
            self.nv2parameter[[self.maxNum-num, self.maxNum+num], i] = \
                self.nv2parameter[[self.maxNum-num, self.maxNum+num], i] / 2
            self.n3parameter[[self.maxNum-num, self.maxNum+num], i] = \
                self.n3parameter[[self.maxNum-num, self.maxNum+num], i] / 2

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, density):
        self._density = density
        self.updateN()

    def updateN(self):

        diameter = np.array([self.fluid["diameter"]]).T

        densityMatrix = self.densityIntegrate(self._density, self.fluid["component"], self.maxNum, self.system["grid"])

        n2 = np.pi * diameter * np.sum(densityMatrix * self.n2parameter, axis = 0)
        n3 = np.pi * np.sum(densityMatrix * self.n3parameter, axis = 0)
        nv2 = 2 * np.pi * np.sum(densityMatrix * self.nv2parameter, axis = 0)

        self.n3 = n3

        n1 = n2 / (2*np.pi * diameter)
        n0 = n2 / (np.pi * diameter**2)
        nv1 = nv2 / (2*np.pi * diameter)

        self.n[0,:] = np.sum(n0, axis = 0)
        self.n[1,:] = np.sum(n1, axis = 0)
        self.n[2,:] = np.sum(n2, axis = 0)
        self.n[3,:] = np.sum(n3, axis = 0)
        self.n[4,:] = np.sum(nv1, axis = 0)
        self.n[5,:] = np.sum(nv2, axis = 0)

    def exChemicalPotential(self):

        diameter = np.array([self.fluid["diameter"]]).T

        dphi = np.zeros((6, self.system["grid"]))
        t = (self.n[3,:] > 10e-5) & ( (1 - self.n[3,:]) > 10e-5 )
        if not t.size > 0:
            print("all packing fraction is less than 10e-5, FMT line 116")
            sys.exit(-1)

        dphi[0,t]= - np.log(1-self.n[3,t])
        dphi[1,t]= self.n[2,t] / (1.0-self.n[3,t])

        dphi[2,t]= (np.log(1-self.n[3,t]) / self.n[3,t] + 1/(1-self.n[3,t])**2)
        dphi[2,t]*= (self.n[2,t]**2 - self.n[5,t]**2) / (12*np.pi*self.n[3,t])
        dphi[2,t]+= self.n[1,t]/(1-self.n[3,t])

        dphi[3,t]= np.log(1-self.n[3,t])/(18*np.pi*self.n[3,t]**3)
        dphi[3,t]+= (1-3*self.n[3,t]+(1-self.n[3,t])**2) / (36*np.pi*self.n[3,t]**2*(1-self.n[3,t])**3)
        dphi[3,t]*= - (self.n[2,t]**3 - 3*self.n[2,t]*self.n[5,t]**2)
        dphi[3,t]+= self.n[0,t]/(1-self.n[3,t]) + (self.n[1,t]*self.n[2,t] - self.n[4,t]*self.n[5,t])/(1-self.n[3,t])**2

        dphi[4,t]= - self.n[5,t] / (1-self.n[3,t])

        dphi[5,t]= np.log(1-self.n[3,t])/self.n[3,t] + 1/(1-self.n[3,t])**2
        dphi[5,t]*= - self.n[2,t]*self.n[5,t] / (6*np.pi*self.n[3,t])
        dphi[5,t]-= self.n[4,t] / (1-self.n[3,t])

        # test_bool = self.n[3,:] < 10e-5
        # dphi[:, test_bool] = 0

        dphiMatrix = self.densityIntegrate(dphi, 6, self.maxNum, self.system["grid"])

        phi = np.zeros((6, self.fluid["component"], self.system["grid"]))
        phi[0,:,:] = np.sum(dphiMatrix[:,0,:].reshape(-1,1,self.system["grid"]) \
                                                      * self.n2parameter, axis=0) / diameter
        phi[1,:,:] = np.sum(dphiMatrix[:,1,:].reshape(-1,1,self.system["grid"]) \
                                                      * self.n2parameter, axis=0) / 2
        phi[2,:,:] = np.pi*diameter * np.sum(dphiMatrix[:,2,:].reshape(-1,1,self.system["grid"]) 
                                                      * self.n2parameter, axis = 0)
        phi[3,:,:] = np.pi * np.sum(dphiMatrix[:,3,:].reshape(-1,1,self.system["grid"]) \
                                                      * self.n3parameter, axis = 0)
        phi[4,:,:] = np.sum(dphiMatrix[:,4,:].reshape(-1,1,self.system["grid"]) \
                                                      * (self.nv2parameter), axis = 0)/(diameter)
        phi[5,:,:] = 2*np.pi * np.sum(dphiMatrix[:,5,:].reshape(-1,1,self.system["grid"]) \
                                                      * (self.nv2parameter), axis = 0)

        phi[[4,5],:,:] = - phi[[4,5],:,:]

        exChemicalPotential = np.sum(phi, axis = 0)
        return exChemicalPotential

    @property
    def freeEnergyDensity(self):
        phi = np.log(1-self.n[3,:]) / (36*np.pi*self.n[3,:]**2) + 1/(36*np.pi*self.n[3,:]*(1-self.n[3,:])**2)
        phi *= self.n[2,:]**3
        phi += -self.n[0,:] * np.log(1-self.n[3,:]) + self.n[1,:]*self.n[2,:]/(1-self.n[3,:])

        return phi/self.n[0,:]

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import BMCSL_EOS
    import Vext

    def testFMT(fluid, system):

        test = FMT1d(fluid, system)

        learningRate = 0.01
        bulkDensity = np.array([system["bulkDensity"]]).T
        gridWidth = system["size"] / system["grid"]
        eos = BMCSL_EOS.BMCSL(fluid)

        exChemicalPotentialBulk = np.zeros((fluid["component"], system["grid"]))
        exChemicalPotentialBulk += np.array(eos.getExChemicalPotential(system["bulkDensity"])).reshape(fluid["component"], 1)

        VextP = Vext.Vext(fluid, system)
        vext = VextP.hardWall()

        density = np.zeros((fluid["component"], system["grid"]))
        density = density * np.exp(-vext)

        error = 1000
        while error > 10e-9:
            oldDesnity = density.copy()
            exChemP = test.getExChemicalPotential(density)

            density = bulkDensity*np.exp(-vext - exChemP + exChemicalPotentialBulk)

            density = learningRate * density + (1-learningRate) * oldDesnity
            error = np.max( np.abs(density - oldDesnity) )
            # print("system evolved")
            # print("error is equal to "+str(test.error))

        return density

    # ============= test H = 3.5 sigma ===============
    # -------------   one  component   ---------------
    def testFMT1C35():
        fluid = {}
        fluid["component"] = 1
        fluid["type"] = "HS"
        fluid["diameter"] = np.array([1.0])
        fluid["wave"] = [1.0]

        system = {}
        system["grid"] = 140
        system["bulkDensity"] = np.array([0.497633])

        system["boundaryCondition"] = "slitPore"
        system["size"] = 3.5

        density = testFMT(fluid, system)

        data = np.loadtxt("./Comparison/FMT1d/slitporeHS3.5.dat")
        c = np.linspace(0,system["size"],system["grid"])

        plt.figure()
        plt.plot(data[:,0], data[:,1])
        plt.plot(c,density.reshape(-1,1))
        plt.savefig('./Comparison/FMT1d/testHS3.5.jpg')
    # -------------   one  component, done   ---------------

    # -------------   two  component   ---------------
    def testFMT2C35():
        fluid = {}
        fluid["component"] = 2
        fluid["type"] = "HS"
        fluid["diameter"] = np.array([1.0, 1.0])
        fluid["wave"] = [1.0, 1.0]

        system = {}
        system["grid"] = 140
        system["bulkDensity"] = np.array([0.497633/2, 0.497633/2])

        system["boundaryCondition"] = "slitPore"
        system["size"] = 3.5

        density = testFMT(fluid, system)
        data = np.loadtxt("./Comparison/FMT1d/slitporeHS3.5.dat")
        c = np.linspace(0,system["size"],system["grid"])

        plt.figure()
        plt.plot(data[:,0], data[:,1])
        density = np.sum(density, axis=0)
        plt.plot(c,density.reshape(-1,1))
        plt.savefig('./Comparison/FMT1d/testHS3.5_2c.jpg')
    # -------------   two  component, done   ---------------
    # ============= test H = 3.5 sigma, done ===============

    # ============= test H = 7.0 sigma ===============
    def testFMT1C70():
        fluid = {}
        fluid["component"] = 1
        fluid["type"] = "HS"
        fluid["diameter"] = np.array([1.0])
        fluid["wave"] = [1.0]

        system = {}
        system["grid"] = 280
        system["bulkDensity"] = np.array([0.434211])
        system["size"] = 7
        system["boundaryCondition"] = "slitPore"

        density = testFMT(fluid, system)
        data = np.loadtxt('./Comparison/FMT1d/slitporeHS7.0.dat')
        c = np.linspace(0,system["size"],system["grid"])

        plt.figure()
        plt.plot(data[:,0], data[:,1])
        plt.plot(c,density.reshape(-1,1))
        plt.savefig('./Comparison/FMT1d/testHS7.0.jpg')
    # ============= test H = 7.0 sigma ===============

    # ============= test H = 30.0 sigma ===============
    def testFMT1C300():
        fluid = {}
        fluid["component"] = 1
        fluid["type"] = "HS"
        fluid["diameter"] = np.array([1.0])
        fluid["wave"] = 1.0

        system = {}
        system["grid"] = 3000
        system["bulkDensity"] = np.array([0.434211])
        system["size"] = 30
        system["boundaryCondition"] = "slitPore"

        density = testFMT(fluid, system)

        c = np.linspace(0,system["size"],system["grid"])
        t = np.ones((system["grid"],1)) * system["bulkDensity"]

        plt.figure()
        plt.plot(c,t)
        plt.plot(c,density.reshape(-1,1))
        plt.savefig('./Comparison/FMT1d/testHS30.0.jpg')
    # ============= test H = 30 sigma, done ===============

    # testFMT1C35()
    # testFMT2C35()
    # testFMT1C70()
    # testFMT1C300()


    # ==================== test the chemical potential for MSA ==================== #
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
    system["grid"] = 250
    system["bulkDensity"] = np.array([0.29, 0.29])

    system["boundaryCondition"] = "slitPore"
    system["size"] = 5.0
    system["cutoff"] = np.array([1.3])
    system["wall"] = "HS"

    system["WallVolt"] = [-0.3, 0.3]

    test = FMT1d(fluid, system)

    f = np.loadtxt("./Comparison/ion/Final_equilibrium.dat")
    f1 = np.loadtxt("./Comparison/ion/chemPFMT_J.dat")
    density = np.zeros((fluid["component"], system["grid"]))
    density[:, 25:226] = f[:, 1:3].T

    test.density = density
    d2 = test.exChemicalPotential()
    d3 = np.linspace(0, system["size"], system["grid"])

    plt.figure()
    plt.plot(f1[:,0]+0.5, f1[:,1], "o")
    plt.plot(d3,d2[0,:])

    plt.plot(f1[:,0]+0.5, f1[:,2], "*")
    plt.plot(d3,d2[1,:], "r")
    plt.savefig("./Comparison/ion/testChemPFMT.jpg")

    # =========-======== test the chemical potential for MSA,done ================= #







