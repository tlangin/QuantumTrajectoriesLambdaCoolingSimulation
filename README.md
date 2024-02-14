# QuantumTrajectoriesLambdaCoolingSimulation
Code that uses a quantum trajectories approach to simulate Lambda cooling in $^{2}\Sigma$ molecules

# Brief Summary
A brief introduction to the code and how to use it can be found in **QTSimulationQuickWriteup**

Even briefer: This code uses a quantum trajectories approach (see Chapter 6 of https://lukin.physics.harvard.edu/files/lukin/files/physics_285b_lecture_notes.pdf) to simulate $\Lambda$ cooling of molecules with a 'typical' $^{2}\Sigma$ level structure.  This assumes cooling on the X-->A transition (though I'd like to add X-->B transitions at some point...), and that the A state has no hyperfine structure.   I've preloaded options to apply this to SrF, CaF, and CaOH.  I can add others upon request.  

Wavefunctions, velocities, and positions of molecules interacting with the polarization-interference-pattern created by 3 counter-propagating pairs of $\sigma^{+}-\sigma^{-}$ lasers (Here I assume a MOT-like configuration, where, along +x and +y, light is $\sigma^{+}$ and along +z it is $\sigma^{-}$.  Vice versa for -x,-y, and-z).  The intensity of the lasers (both overall intensity and the ratio between the two coupled states), what pair of states are being "\Lambda" coupled, the detuning (both overall detuning and raman detuning), are all user inputs.

# QTSimulationQuickWriteup

I strongly recommend any user to read this pdf before running simulations.  It explains what the code does (in light detail) and shows some results.

# QTOBESimulationWriteupMoreDetails

Has some more details about the code, along with some more benchmarking, primarily of results from this paper by Mike Tarbutt: https://arxiv.org/abs/1608.04645

# mainOBEWriteup

This is a writeup primarily for another code on this repository (see: OBESimulation), but it's relevant here as the appendix contains the calculation for the `clebsch-gordan' like terms in the coupling matrices ('cs') used in this code.

# lambdaCoolingFinalAddCaOH.cpp

The simulation is all contained in this c++ code.  It uses the Armadillo (http://arma.sourceforge.net/download.html) library, which you will need to download if you are planning on running this on your home computer (alternately, most `super-computing' clusters already have this installed, and then you can just load it as a module).  More details are in the **QTSimulationQuickWriteup**.

# testBatch.sbatch

Batch file I've used to compile and submit the simulation to the cluster.  Note the line:

./testVaryCaFSrFNoPathLengthCorrectPol 2.9 \$dRaman $satParam 0.9 0 2 0

This runs the executable file created by the compiler (here, testVaryCaFSrFNoPathLengthCorrectPol, but you can change the name of the executable in the g++ compilation line).  This simulation requires 7 inputs:

1) Detuning (here, 2.9).  This is the overall laser detuning ($\Delta$ in figure 1 of **QTSimulationQuickWriteup**)

2) deltaRaman (here dRaman, which is iterated over in the batch script).  This is the `raman' detuning ($\delta_{R}$ in figure 1 of **QTSimulationQuickWriteup**)

3) saturation parameter.  This is the total saturation parameter ($I/I_{Sat}$ where $I_{sat}= \frac{\hbar c\Gamma\omega^{3}}{4\omega\times 3c^{3}}$) of **one** pass of the laser (includes both frequency components of the beam)

4) $R_{21}$ (here 0.9).  This is the power ratio between the two frequency components $I_{2}/I_{1}$.

5) First state (here 0, which is defined to be $|F=1,J=1/2\rangle$ state, almost always the case for $\Lambda$ cooling).  State 1 is $|F=0,J=1/2\rangle$, state 2 is $|F=1,J=3/2\rangle$ and state 4 is $|F=2,J=3/2\rangle$ (this is in increasing energy order for all $^{2}\Sigma, N=1$ groundstate molecules that I am aware of).

6) Second state (here 2, e.g. $|F=1,J=3/2\rangle$)

7) CaOHOrCaFOrSrF (here 0).  2 = 'use CaOH values' (hyperfine energies in |X,N=1> state, J-Mixing parameter 'a', etc.), 1 = 'use CaF values', 0 = 'use SrF values'.

# testBash.sh

Same as above, except without the supercomputing cluster stuff.  Use this to run on your home computer (warning: will take a good bit of time and use all available CPU since I've enabled parallelization).  I'd only use this for very basic testing, and perhaps even go down to like 1 particle


