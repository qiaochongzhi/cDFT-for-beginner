import numpy as np

class adam(object):

    """This class is used to solve PDE by using Adams-Bash and Adams-Moulton method"""

    def __init__(self, drive, deltaT):

        self.deltaT = deltaT
        self.deriveOld0 = np.array(drive)
        self.deriveOld1 = np.array(drive)
        self.deriveOld2 = np.array(drive)

    def updateDerive(self, drive):

        self.deriveOld2 = self.deriveOld1.copy()
        self.deriveOld1 = self.deriveOld0.copy()
        self.deriveOld0 = drive.copy()

    def adamsBash(self, rho, drive, k):

        if k < 4:
            newDensity = rho + self.deltaT * ( (1.5)*drive - (0.5)*self.deriveOld0 )
        else:
            delta = 55*drive - 59*self.deriveOld0 + 37*self.deriveOld1 - 9*self.deriveOld2
            newDensity = rho + (self.deltaT/24) * delta
        
        return newDensity

    def adamsMoulton(self, rho, drive, newDerive, k):

        if k < 4:
            newDensity = rho + (self.deltaT/2) * ( drive + newDerive )
        else:
            delta = 9*newDerive + 19*drive - 5*self.deriveOld0 + self.deriveOld1
            newDensity = rho + (self.deltaT/24) * delta

        return newDensity

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # deltaT = 0.00001

    # density = np.linspace(0,1,11)

    # def test(x, t):
        # return x*t

    # solve = adam(test(density,0),deltaT)

    # tot = 6000

    # result = np.zeros((11, tot))

    # for i in range(tot):
        # t = i*deltaT

        # drive = test(density, t)
        # # print(drive)
        # newDensity = solve.adamsBash(density, drive, i)
        # for j in range(3):
            # oldDensity = newDensity
            # newDerive = test(newDensity, t+deltaT)
            # newDensity = solve.adamsMoulton(density, drive, newDerive, i)
            # testvalue = np.abs(np.sum(oldDensity) - np.sum(newDensity))/np.sum(oldDensity)
            # if testvalue < 0.05:
                # # print("break")
                # break

        # newDerive = test(newDensity, t+deltaT)
        # solve.updateDerive(newDerive)
        # density = newDensity.copy()

        # # print("new density")
        # # print(newDensity)
        # result[:,i] = newDensity.reshape((11))


    #=========== Euler Method ==========

    # result2 = np.zeros((11, tot))
    # d = np.linspace(0,1,11)
    # for i in range(tot):
        # newD = d + deltaT * test(d, i*deltaT)
        # result2[:,i] = newD.reshape((11))
        # d = newD

    # ==========Euler Method End==========


    # x = np.linspace(0, 1, 11)
    # y = x * np.exp(0.5 * (tot * deltaT)**2)
    # plt.figure()
    # plt.plot(x, result[:,-1], "r^")
    # plt.plot(x, result2[:,-1], "o")
    # plt.plot(x, y)
    # # plt.show()
    # plt.savefig('./testABAM.jpg')

    #===================test=====================

    from scipy.integrate import odeint

    def diff_equation(y_list, x):
        y,z = y_list
        return np.array([z,-y])

    x=np.linspace(0,np.pi*2,num=100)
    y0=[1,1]
    result=odeint(diff_equation,y0,x)

    #============================================================

    deltaT = 0.01

    density = np.array([1,1])

    solve = adam(diff_equation(density,0),deltaT)

    tot = int(6.28/deltaT)

    result1 = np.zeros((2, tot))

    for i in range(tot):
        t = i*deltaT

        drive = diff_equation(density, t)
        # print(drive)
        newDensity = solve.adamsBash(density, drive, i)
        for j in range(3):
            oldDensity = newDensity
            newDerive =diff_equation(newDensity, t+deltaT)
            newDensity = solve.adamsMoulton(density, drive, newDerive, i)
            testvalue = np.abs(np.sum(oldDensity) - np.sum(newDensity))/np.sum(oldDensity)
            if testvalue < 0.05:
                # print("break")
                break

        newDerive = diff_equation(newDensity, t+deltaT)
        solve.updateDerive(newDerive)
        density = newDensity.copy()

        # print("new density")
        # print(newDensity)
        result1[:,i] = newDensity.reshape((2))
    #============================================================

    xx = np.linspace(0, np.pi*2, num=tot)

    plt.plot(x,result[:,0],label='y')
    plt.plot(x,result[:,1],label='z')
    plt.plot(xx.T,result1[0,:]+0.02,label='yy', linestyle = ":")
    plt.plot(xx.T,result1[1,:]+0.02,label='zz', linestyle = "-.")
    plt.legend()
    plt.grid()
    plt.savefig("./testADAMnew.jpg")


    # def lorenz(w, t, p, r, b):

        # x, y, z = w

        # return np.array([p*(y-x), x*(r-z)-y, x*y-b*z])

    # t = np.arange(0, 30, 0.01)

    # track1 = odeint(lorenz, (0.0, 1.00, 0.0), t, args=(10.0, 28.0, 3.0))
    # track2 = odeint(lorenz, (0.0, 1.01, 0.0), t, args=(10.0, 28.0, 3.0))

    # from mpl_toolkits.mplot3d import Axes3D
    
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot(track1[:,0], track1[:,1], track1[:,2])
    # ax.plot(track2[:,0], track2[:,1], track2[:,2])
    # plt.savefig("./testADAMnew.jpg")

