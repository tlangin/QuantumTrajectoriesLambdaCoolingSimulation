#!/usr/bin/env bash
detRamans=(-0.12 -0.08 -0.04 0.0 0.04 0.08 0.12);
satParams=(1 3 5 7 9);
g++ lambdaCoolingFinalAddCaOH.cpp -o test -larmadillo -O3 -fopenmp
#CaF scan like Fig. 2a of CaF paper on Lambda Imaging.  Det=+2.9Gamma, R_21 = 0.9, states coupled
#are |F=1,J=1/2> (0) and |F=2,J=3/2> (3)
for satParam in ${satParams[@]}; do 
   for dRaman in ${detRamans[@]}; do
       ./test 2.9 $dRaman $satParam 0.9 0 3 1
       done
done

#Same as above but for |F=1,J=1/2> (0) and |F=1,J=3/2> (2)
for satParam in ${satParams[@]}; do 
   for dRaman in ${detRamans[@]}; do
       ./test 2.9 $dRaman $satParam 0.9 0 2 1
       done
done

#now repeat for SrF (last variable now 0)
for satParam in ${satParams[@]}; do 
   for dRaman in ${detRamans[@]}; do
       ./test 2.9 $dRaman $satParam 0.9 0 3 0
       done
done

#Same as above but for |F=1,J=1/2> (0) and |F=1,J=3/2> (2)
for satParam in ${satParams[@]}; do 
   for dRaman in ${detRamans[@]}; do
       ./test 2.9 $dRaman $satParam 0.9 0 2 0
       done
done

