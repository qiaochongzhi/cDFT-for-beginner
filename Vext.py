import numpy as np

class Vext():

    def __init__(self, fluid, system):

        self.fluid = fluid
        self.system = system

        self.gridWidth = self.system["size"] / self.system["grid"]

        self.vext = np.zeros((self.fluid["component"], self.system["grid"]))


    def hardWall(self):

        for i in range(self.fluid["component"]):

            grid = round(self.fluid["diameter"][i] / (2*self.gridWidth))
            grid = int(grid)

            self.vext[i,:grid] = np.inf
            if self.system["boundaryCondition"] == "slitPore":
                self.vext[i,-(grid):] = np.inf

        return self.vext

    def grapheneWall(self):

        def steelWall1(r, sigma, epsilon, temperature):

            sigma_w = 3.4
            epsilon_w = 0.2328049
            delta_w = 3.34
            rho_w = 0.114

            # sigma_aw = sigma
            # epsilon_aw = 6.283 * epsilon
            # delta_w = 0.7071 * sigma

            r = np.abs(r)
            sigma_aw = (sigma_w + sigma)/2
            epsilon_aw = np.sqrt(epsilon * epsilon_w)

            u = 2*np.pi*rho_w * delta_w * epsilon_aw * sigma_aw**2

            u = epsilon_aw

            u *= ((2./5.)*(sigma_aw/r)**10 - (sigma_aw/r)**4 - (sigma_aw**4/(3*delta_w*(r+0.61*delta_w))**3))
            u *= temperature

            return u

        for i in range(self.fluid["component"]):

            # grid1 = round(self.system["cutoff"][0]/self.gridWidth)
            # grid1 = int(grid1)

            grid2 = round(self.system["size"]/self.gridWidth)
            grid2 = int(grid2)

            # grid = min(grid1, grid2)
            grid = grid2

            sigma = self.fluid["sigma"][i]
            epsilon = self.fluid["epsilon"][i]
            temperature = self.fluid["temperature"]
            self.vext[i, :grid] += [steelWall1(self.gridWidth*(x+0.5), sigma, epsilon, temperature) for x in range(grid)]
            if self.system["boundaryCondition"] == "slitPore":
                self.vext[i, -grid:] += [steelWall1(self.gridWidth*(np.abs(x)+0.5), sigma, epsilon, temperature) for x in range(grid,0,-1)]


        return self.vext

    def ljWall(self):

        def steelWall(r, sigma, epsilon, temperature):

            # sigma_w = 3.4
            # epsilon_w = 0.2328049
            # delta_w = 3.34
            # rho_w = 0.114

            sigma_aw = sigma
            epsilon_aw = 6.283 * epsilon
            delta_w = 0.7071 * sigma

            r = np.abs(r)
            # sigma_aw = (sigma_w + sigma)/2
            # epsilon_aw = np.sqrt(epsilon * epsilon_w)

            # u = 2*np.pi*rho_w * delta_w * epsilon_aw * sigma_aw**2

            u = epsilon_aw

            u *= ((2./5.)*(sigma_aw/r)**10 - (sigma_aw/r)**4 - (sigma_aw**4/(3*delta_w*(r+0.61*delta_w))**3))
            u /= temperature
            # u *= temperature

            return u

        for i in range(self.fluid["component"]):

            # grid1 = round(self.system["cutoff"][0]/self.gridWidth)
            # grid1 = int(grid1)

            grid2 = round(self.system["size"]/self.gridWidth)
            grid2 = int(grid2)

            # grid = min(grid1, grid2)
            grid = grid2

            sigma = self.fluid["sigma"][i]
            epsilon = self.fluid["epsilon"][i]
            temperature = self.fluid["temperature"]
            self.vext[i, :grid] += [steelWall(self.gridWidth*(x+0.5), sigma, epsilon, temperature) for x in range(grid)]
            if self.system["boundaryCondition"] == "slitPore":
                self.vext[i, -grid:] += [steelWall(self.gridWidth*(np.abs(x)+0.5), sigma, epsilon, temperature) for x in range(grid,0,-1)]

            # a = self.vext > 20
            # self.vext[a] = np.inf

        return self.vext


    def perturbWall(self, ep):

        L = 15/2 + 1.5
        T = 1
        A = 10
        m = 20
        epsilon = ep
        z0 = 20.48/2

        w = 3/2

        grid = self.system["grid"]

        def pWall(z, Li):

            V = A * ( (z-z0)/Li )**m + epsilon * np.exp( -( (z-z0)/w )**2 )

            return V

        for i in range(self.fluid["component"]):

            Li = L - self.fluid["diameter"][i]/2

            self.vext[i, :] += [pWall(x*self.gridWidth, Li) for x in range(grid)]

        return self.vext




    @property
    def squareWell(self):

        pass
        # for i in range(self.fluid["component"]):

            # gird = round(self.fluid["diameter"][i] / (2*self.gridWidth))
            # gird = int(grid)

            # self.vext[i,:grid] = np.inf
            # self.vext[i,-(grid):] = np.inf

        # return self.vext


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    fluid = {}
    fluid["component"] = 1
    fluid["sigma"] = np.array([1.0])
    fluid["epsilon"] = np.array([1.0])
    fluid["temperature"] = 1.0

    system = {}
    system["grid"] = 350
    system["cutoff"] = np.array([5.0])
    system["size"] = 7.0
    system["boundaryCondition"] = "slitPore"

    # system["wall"] = "PW"
    # system["epsilon"] = -2

    test = Vext(fluid, system)

    vext = test.ljWall()
    b = vext > 10
    vext[b] = 10

    a = np.linspace(0, system["size"], system["grid"])
    plt.figure()
    plt.plot(a, vext.T)
    plt.savefig("./Comparison/testVext/testVext.jpg")

    system["grid"] = 400
    system["cutoff"] = np.array([3.0])
    system["size"] = 4.0
    test = Vext(fluid, system)

    vext = test.ljWall()
    b = vext > 10
    vext[b] = 10

    a = np.linspace(0, system["size"], system["grid"])
    plt.figure()
    plt.plot(a, vext.T)
    plt.savefig("./Comparison/testVext/testVext4.0.jpg")
