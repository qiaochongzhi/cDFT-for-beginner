# cDFT-for-beginner
This is a project of 1D classical density functional theory for beginner

## Framework

This is a ongoing project, it contains a hard-sphere or a Lennard-Jones fluid confined in a slit pore or around a single wall.
This project is to help the beginner to understand the framework of classical density functional theory.
I strongly recommend you read the code carefully, before you use it to do any calculation. 
Before use it, please make sure 'scipy', 'numpy', 'matplotlib' and 'numba' have been installed.

In the following, I will introduce all programme in this project:
* 'dft1d.py': this is the core function which contains the framework of this project.
* 'MBWR_EOS.py': the equation of state for Lennard-Jones fluids proposed by Keith E. Gubbins (Molecular Physics, 1993, 78(3), 591-618).
* 'BMCSL_EOS.py': the equation of state for hard-sphere fluids (J. Chem. Phys. 1971 54, 1523).
* 'Functional.py': define some general useful functions for calculating the chemical potential.
* 'FMT1d.py': calculate the excess chemcial potential of hard-sphere interaction proposed by Rosenfeld in 1989, and improved by Prof. Wu or Prof. Roth. (PRL 1989)
* 'MFA1d.py': calculate the excess chemcial potential of attractive part of Lennard-Jones potential by using mean field theory. *do not use correlation functional ('corChemicalPotential')*
* 'FMSA1d.py': calculate the excess chemical potential of the attractive part of Lennard-Jones potential by using first-order mean spherical approximation proposed by Y. Tang.
* 'Vext.py': calculate the external potential, contains hard wall and 10-4-3 wall.
* 'ADAM.py': the ADAM algorithm for solving an ordinary differential equation, however, rk45 in scipy is faster and more robust, which means I do not recommend to use it.
* 'DDFT.py': the programme for dynamical density functional theory which inherits from the 'dft1d.py'.
* "MSA1d.py": **this programme contains some mistake, do not use it.**

## unit 

Here, I will introduce all unit used in this programme.

### the reduced temperature

$\beta   = 1 / (Kb * Na * T)$
$[\beta] = kJ / mol$
$kb = 1.38 * 10^(-23) J/K$
$Na = 6.02 * 10^23 (1/mol)$
$kb * Na / 1000 = 0.0083076$

which means when $\beta$ is equal to one, the temperature is equal to $120.371K (1/0.0083076)$.

### the unit of density and diameter

In fact, the packing fraction is a dimensionless number, whic is equal to $\frac{\pi}{6} \rho \sigma^3$. 
However, for 10-4-3 wall, the diameter of carbon is about $3.4A$, for convenience, you can consider the unit of length is A, 
and the unit of density is $1/A^3$.

### the unit of charge fluid

First, we consider the unit of the Bjerrum length , $l_b$. from its definition, $\beta e^2 / 4 \pi \epsilon_0 \epsilon_r$,
where $e = 1.602 \times 10^{-19} C$ is the elementary charge, the $\epsilon_0 = 8.854\times 10^{-12} F/m$ is the Vaccum permittivity,
and $\epsilon_r$ is the relative permittivity, which is a dimensionless number.

Hence, the reduced Bjerrum length is equal to 

$
\beta l_b / \sigma_L = \frac{ (1.6 \times 10^{-19})^2 }{ 4 \pi (1.38 \times 10^{-23}) T (8.854 \times 10^{-12}) \sigma_L }.
$

As we mentioned before, the above equation can be written as 

$
\beta l_b / \sigma_L = \frac{ (1.6 \times 10^{-19})^2 \times 6.02 \times 10^{23} / 1000 }{ 4 \pi (8.854 \times 10^{-12}) T^* \sigma_L },
$

The $\sigma_L$ is the unit length which is assumed as $5.0\times10^{-10}m$

