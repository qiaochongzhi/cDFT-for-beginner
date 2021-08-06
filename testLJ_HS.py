import numpy as np
import dft1d

import time


import matplotlib.pyplot as plt

def testDFT(fluid, system):

    test = dft1d.dft1d(fluid, system)

    while test.error > 10e-5:
        test.evolveSystem()

    return test.density

fluid = {}
fluid["component"] = 1
# fluid["type"] = "LJ"
fluid["type"] = "HS"
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
        file1 = './Comparison/LJ_cDFT/testHS_LJ' + str(system["size"]) + s + '_FMSA.jpg'
    else:
        if system["Cor"]:
            file1 = './Comparison/LJ_cDFT/testLJ' + str(system["size"]) + s + '_MFA_Cor.jpg'
        else:
            file1 = './Comparison/LJ_cDFT/testLJ' + str(system["size"]) + s + '_MFA.jpg'
    plt.savefig(file1)

def testDifferentMethod(fluid, system, label):
    # system["LJ"] = "MFA"

    # system["Cor"] = True
    # plotResult(fluid, system, label)

    # system["Cor"] = False
    # plotResult(fluid, system, label)

    system["LJ"] = "FMSA"
    plotResult(fluid, system, label)


# ==================== test slit pore system ==================== #

'''
The simulation data can be found in Y. Tang et al. PRE, 70, 011201
(2004) Figure 6 and Figure 7.
'''

start = time.time()

# --------------------   test H = 7.5 sigma  -------------------- #
# system["grid"] = 750
# system["cutoff"] = np.array([5.0])


system["size"] = 7.5
system["grid"] = 375
system["cutoff"] = np.array([3.0])

testDifferentMethod(fluid, system, 1)

# --------------------   test H = 4.0 sigma  -------------------- #
system["size"] = 4.0
system["grid"] = 200
system["cutoff"] = np.array([3.0])

testDifferentMethod(fluid, system, 1)

# ================= test slit pore system, done ================= #

fluid["type"] = "LJ"

fluid["temperature"] = 2.4

system["wall"] = "HS"

system["grid"] = 140
system["bulkDensity"] = np.array([0.497633])
system["boundaryCondition"] = "slitPore"
system["size"] = 3.5
system["LJ"] = "FMSA"

density = testDFT(fluid, system)

data = np.loadtxt("./Comparison/FMT1d/slitporeHS3.5.dat")
c = np.linspace(0, system["size"], system["grid"])

plt.figure()
plt.plot(data[:,0], data[:,1])
plt.plot(c, density.reshape(-1, 1))
plt.savefig("./Comparison/LJ_cDFT/testHS_LJ3.5_FMSA.jpg")


print(time.time() - start)

