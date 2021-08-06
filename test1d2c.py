import numpy as np
import sys
import dft1d

import matplotlib.pyplot as plt

import time

fluid = {}
fluid["component"] = 2
fluid["type"] = "LJ"
fluid["sigma"] = np.array([1.0, 1.0])
fluid["wave"] = [1.0, 1.0]
fluid["temperature"] = 1.2
fluid["epsilon"] = np.array([1.0, 1.0])

system = {}
# system["grid"] = 1500
system["grid"] = 375
system["bulkDensity"] = np.array([0.5925/2, 0.5925/2])

system["boundaryCondition"] = "slitPore"
system["size"] = 7.5
system["cutoff"] = np.array([3.0])
system["wall"] = "LJ"

def testDFT(fluid, system):

    test = dft1d.dft1d(fluid, system)

    while test.error > 10e-5:
        test.evolveSystem()

    return test.density

def plotResult(fluid, system):

    file = "./Comparison/LJ_cDFT/slitporeLJ7.5.dat"

    a = np.linspace(0, system["size"], system["grid"])
    c = np.loadtxt(file)
    t = testDFT(fluid, system)
    t = np.sum(t, axis=0)

    plt.figure()
    plt.xlim((0,system["size"]))
    plt.ylim((0,4.5))
    plt.plot(c[:,0], c[:,1], linestyle="None", marker="o")
    plt.plot(a, t.T)
    if system["LJ"] == "FMSA":
        file1 = './Comparison/LJ_cDFT/testLJ' + str(system["size"]) + '_FMSA_2c.jpg'
    else:
        if system["Cor"]:
            file1 = './Comparison/LJ_cDFT/testLJ' + str(system["size"]) + '_MFA_Cor_2c.jpg'
        else:
            file1 = './Comparison/LJ_cDFT/testLJ' + str(system["size"]) + '_MFA_2c.jpg'
    plt.savefig(file1)

def testDifferentMethod(fluid, system):

    system["LJ"] = "MFA"
    # system["Cor"] = True
    # plotResult(fluid, system)

    system["Cor"] = False
    plotResult(fluid, system)

    system["LJ"] = "FMSA"
    plotResult(fluid, system)

start = time.time()

testDifferentMethod(fluid, system)

print(time.time() - start)

