#!/bin/bash
#SBATCH --job-name=test_sbatch
#SBATCH --time=24:00:00
#SBATCH --partition=broadwl
#SBATCH -c 20
#SBATCH --mail-user=tlangin@uchicago.edu
module load armadillo/7.8
detRamans=(-0.1 -0.02 0.0 0.02 0.10);
satParams=(1.5 3 8);
g++ -std=c++11 lambdaCoolingFinalAddCaOH.cpp -o testVaryCaFSrFNoPathLengthCorrectPol -larmadillo -O3 -fopenmp
#CaF scan like Fig. 2a of CaF paper on Lambda Imaging.  Det=+2.9Gamma, R_21 = 0.9, states coupled
#are |F=1,J=1/2> (0) and |F=2,J=3/2> (3)
#for satParam in ${satParams[@]}; do 
#   for dRaman in ${detRamans[@]}; do
#       ./testVaryCaFSrFNoPathLengthCorrectPol 2.9 $dRaman $satParam 0.9 0 3 1
#       done
#done

#Same as above but for |F=1,J=1/2> (0) and |F=1,J=3/2> (2)
#for satParam in ${satParams[@]}; do 
#   for dRaman in ${detRamans[@]}; do
#       ./testVaryCaFSrFNoPathLengthCorrectPol 2.9 $dRaman $satParam 0.9 0 2 1
#       done
#done

#now repeat for SrF (last variable now 0)
#for satParam in ${satParams[@]}; do 
#   for dRaman in ${detRamans[@]}; do
#       ./testVaryCaFSrFNoPathLengthCorrectPol 2.9 $dRaman $satParam 0.9 0 3 0
#       done
#done

#Same as above but for |F=1,J=1/2> (0) and |F=1,J=3/2> (2)
for satParam in ${satParams[@]}; do 
   for dRaman in ${detRamans[@]}; do
       ./testVaryCaFSrFNoPathLengthCorrectPol 2.9 $dRaman $satParam 0.9 0 2 0
       done
done

#detRamansTwo=(-.06 -.04 -.02  0.0 .02 .04 .06 .08 .1);
#CaF scan from Doyle Paper
#for dRaman in ${detRamansTwo[@]}; do
#   ./testSmallerTimeStep 2.9 $dRaman 2.8 0.9 0 3 1
#done
