import numpy as np

def compressibility(density, diameter):

  eta = np.pi * density * diameter**2 / 6.0
  Z   = (1 + eta + eta**2 - eta**3)/(1-eta)**3
  return Z

def chemicalPotential(density, diameter, wave):
  
  eta = np.pi * density * diameter**2 / 6.0
  mu  = np.log(density * wave**3) + (8*eta - 9*eta**2 + 3*eta**3)/(1-eta)**3
  return mu

def freeEnergy(density, diameter, wave):

  eta = np.pi * density * diameter**2 / 6.0
  F   = np.log(density * wave**3) - 1 + 4*(eta/(1-eta)) + (eta/(1-eta))**2
  return F

if __name__ =="__main__":
  print("-------test pressure---------")
  print("compressibility from eos " + str(compressibility(0.4, 1)))
  print("compressibility is equal to 2.518")
  print("-------test chemical potential---------")
  print("chemical potential from eos " + str(chemicalPotential(0.4, 1, 1)))
  print("chemical potential is equal to 1.7316")
  print("-------test free energy---------")
  print("free energy from eos " + str(freeEnergy(0.4, 1, 1)))
  print("free energy is equal to -0.786404")
