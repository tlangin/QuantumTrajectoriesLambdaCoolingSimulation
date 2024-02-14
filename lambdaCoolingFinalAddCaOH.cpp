//TO DO

//1) Add ability to pass arguments detRaman, intensity (also, norm to ISat instead of IRat), R21, second state, etc. instead of "job"
//2) make savefolder simpler (saveData/(userHeader)...etc.)
//3) make generic matlab folder for analysis of histograms.
//4) force N=1, start v=0?

/***********************************************************************/
/*                                                                     */
/*                                                                     */
/* This code simulates lambda cooling of the SrF molecule              */
/* this all assumes X-->A transition, that '0' laser energy corresponds*/
/* to the |F=1,J=1/2>-->|F'=1> transition (prime is A state)           */
/* all energies are normalized to \hbar*gamma where                    */
/* gamma is 1/lifetime_A (6.63 MHz)                                    */
/* wavefunction 'order' is |F=1,J=1/2> (1-3),|F=0> (4),|F=1,J=3/2>(5-7)*/
/* |F=2,J=3/2> (8-12), |F'=1,J'=1/2> (15-13), |F'=0,J'=1/2> (16)       */
/* Coupling from ground state to excited states are hardcoded          */
/* in 'main' (end of file)                                             */
/*                                                                     */
/***********************************************************************/

/******************************************************************************************************************************************/
/*                                                                                                                                        */
/*    Compile in linux environment on home computer:                                                                                      */
/*                                                                                                                                        */
/*    g++ LaserCoolWithExpansion.cpp -o runFile -larmadillo -O3 -fopenmp                                                                  */
/*    (the O3 isn't strictly necessary, it just optimizes speed of some mathematical functions)                                           */
/*                                                                                                                                        */
/*    Run on Home computer: ./test detuning detRaman IRat R21 firstState secondState CaFOrSrF                                             */
/*    detRaman is raman detuning in units Gamma, IRat is the saturation parameter of a single laser pass (SrF X->A, ISat = 2.8 mW/cm^2)   */
/*    firstState and secondState are two states addressed (0=|F=1,J=1/2>, 1=|F=0,J=1/2>, 2=|F=1,J=3/2>, 3=|F=2,J=3/2>  ) CaFOrSrF chooses */
/*    between CaF(1) and SrF(0, or default)                                                                                               */ 
/*                                                                                                                                        */
/*                                                                                                                                        */
/******************************************************************************************************************************************/


#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<time.h>
#include<omp.h>
#include<sys/stat.h>
#include<iostream>
#include<complex>
#include<armadillo>
#include<random>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;
using namespace arma;
std::random_device rd;
std::mt19937 rng(rd());
std::uniform_real_distribution<double> uni(0, 1);
auto random_double = uni(rng);
std::complex<double> I(0,1);
/* GLOBAL VARIABLES */

/* save Directory */
char saveDirectory[256] = "saveDataSrFAndCaFNormalTimeStepNoPathLengthCorrectPol/";//main directory.  A subfolder titled  Gamma_Kappa_Number___ will be created.  Subsubfolders for each job will be created.  data will be stored in the job folders;

/*fixed parameters */
double SrFGroundEnergies[4] = {0.0,7.5,19.6,25.9};
double CaFGroundEnergies[4] = {0.0,9.3,15.1,18.1};
double CaOHGroundEnergies[4] = {0.0,0.001,8.006,8.243};

double vKickSrF = 0.0013;//vKick for SrF
double vKickCaF = 0.0022;//CaF
double vKickCaOH = 0.0028;//CaOH;
double vKick;
double vSpread = 0.02;//in units kv/gam.  0.02 = 100uK for SrF, 71 uK for CaF.  This is fine starting point

double SrFAParam = 0.888;
double CaFAParam = 0.772496;
double CaOHAParam = 0.999633;
double aParam;
double groundEnergies[4];

double SrFGamOverC = 0.14;
double CaFGamOverC = 0.17;
double CaOHGamOverC = 0.13;
double gamOverC;

/*input variables: the only ones you'll ever really want to change*/

double tmax =500000;//4000 is around 100 us FYI.  I find this is enough time to get a reasonable sample of the steady state velocity distribution
double retroPathLength = 0.0;//retroPathLength In Meters.  Make this zero if you don't want to account for differential phase shift on retro reflected beams

/*preinitialized stuff */

double detuning;//SP detuning normalized by \gamma_{SP}
double detShift;//adjusts from zero when including only one F state and that state is not F=1Down
double satParam;
double R21;
double Om;
double OmRepump;
double detRaman;
double detLaserDiff;
unsigned firstState;
unsigned secondState;
unsigned CaOHOrCaFOrSrF;
bool applySpontForce=true;
bool applyForce=true;
bool reNormalizewvFns = true;

// some stuff I had put in for testing what happens if decays to some states are eliminated, just ignore these unless, e.g. you want to isolate effect from decay to states not coupled via lambda cooling
bool onlyFg1DownToFe1 = false;
bool onlyFg1DownToFe0 = false;
bool onlyFg1UpToFe1 = false;
bool onlyFg1UpToFe0 = false;
bool onlyFg0ToFe1 = false;
bool onlyFg2ToFe1 = false;

bool removeFg1Down = false;
bool removeFg1Up = false;
bool removeFg0 = false;
bool removeFg2 = false;

/* other input variables: You'll probably never want to change these*/
const int N0 =20;                // number of particles
int nrthread = 20;              // number of threads
int currFluorCounts = 0;
int sampleFreq = 1000;		   // output data for all functions every X timesteps (so, every 10\Gamma record velocities, population in each state, etc.)
#define TIMESTEP .01		   // default time step
double dt;                     // time step
double t;                      // actual time
int c0;
/* output variables */
unsigned counter=0;            // time counter, used as output-file label
/* system variables */
//const double temperature = 0.01; //(Temp in K)
//std::normal_distribution<double> velocityDistribution(0, 1.0508*sqrt(temperature));//set up velocity generator, spread given by sqrt(T_norm)...i.e. sqrt(1/Gamma)
//double V[3][N0+1000];        // ion velocities
//double X[3][N0+1000];        // ion positions
//test with one atom
double V[3][N0];
double X[3][N0];
//double speeds[20];
//double vels[21];//vels to study
//double forceForGivenVel[20000];//should be tmax/timestep/samplefreq long

std::normal_distribution<double> velocityDistribution(0, vSpread);//set up velocity generator, spread given by sqrt(T_norm)...i.e. sqrt(1/Gamma)
//waveFunctions
cx_mat wvFns[N0];
double tPart[N0];
double numStates = 16;
mat ident = mat(numStates,numStates,fill::eye);
cx_mat wvFn1=cx_mat(ident.col(0),mat(numStates,1,fill::zeros));//|F=1,J=1/2,m=-1>
cx_mat wvFn2=cx_mat(ident.col(1),mat(numStates,1,fill::zeros));//|F=1,J=1/2,m=0>
cx_mat wvFn3=cx_mat(ident.col(2),mat(numStates,1,fill::zeros));//|F=1,J=1/2,m=+1>
cx_mat wvFn4=cx_mat(ident.col(3),mat(numStates,1,fill::zeros));//|F=0,J=1/2,m=0>
cx_mat wvFn5=cx_mat(ident.col(4),mat(numStates,1,fill::zeros));//|F=1,J=3/2,m=-1>
cx_mat wvFn6=cx_mat(ident.col(5),mat(numStates,1,fill::zeros));//|F=1,J=3/2,m=0>
cx_mat wvFn7=cx_mat(ident.col(6),mat(numStates,1,fill::zeros));//|F=1,J=3/2,m=+1>
cx_mat wvFn8=cx_mat(ident.col(7),mat(numStates,1,fill::zeros));//|F=2,J=3/2,m=-2>
cx_mat wvFn9=cx_mat(ident.col(8),mat(numStates,1,fill::zeros));//|F=2,J=3/2,m=-1>
cx_mat wvFn10=cx_mat(ident.col(9),mat(numStates,1,fill::zeros));//|F=2,J=3/2,m=0>
cx_mat wvFn11=cx_mat(ident.col(10),mat(numStates,1,fill::zeros));//|F=2,J=3/2,m=+1>
cx_mat wvFn12=cx_mat(ident.col(11),mat(numStates,1,fill::zeros));//|F=2,J=3/2,m=+2>
//(YES the signs below are correct for the F' state see cs, gs terms in main().  I don't know why I (TKL) chose to do it this way where zeeman quantum number is reversed in excited state.
//Probably made sense at the time but if I were doing it over again I'd switch so that wavefunctions are ordered in terms of ascending quantum number for all states, not just the ground...
cx_mat wvFn13=cx_mat(ident.col(12),mat(numStates,1,fill::zeros));//|F'=1,J=1/2,m=+1> (again, note the +1)
cx_mat wvFn14=cx_mat(ident.col(13),mat(numStates,1,fill::zeros));//|F'=1,J=1/2,m=0>
cx_mat wvFn15=cx_mat(ident.col(14),mat(numStates,1,fill::zeros));//|F'=1,J=1/2,m=-1> (note the -1)
cx_mat wvFn16=cx_mat(ident.col(15),mat(numStates,1,fill::zeros));//|F'=0,J=1/2,m=0>


//decay coupling and rates and hamiltonians (as much as you can without the actual particle input...so, everything but the proper values of sig+, sig-, etc.)
cx_mat cs[36];
double gs[36];
cx_mat hamDecayTerm=cx_mat(mat(numStates,numStates,fill::zeros),mat(numStates,numStates,fill::zeros));//now factor in the decay type terms
cx_mat hamEnergyTerm=cx_mat(mat(numStates,numStates,fill::zeros),mat(numStates,numStates,fill::zeros));//now factor in the energy type terms
cx_mat hamCouplingTermOnlySigPlus=cx_mat(mat(numStates,numStates,fill::zeros),mat(numStates,numStates,fill::zeros));//now factor in the coupling type terms
cx_mat hamCouplingTermOnlySigMinus=cx_mat(mat(numStates,numStates,fill::zeros),mat(numStates,numStates,fill::zeros));//now factor in the coupling type terms
cx_mat hamCouplingTermOnlyPi=cx_mat(mat(numStates,numStates,fill::zeros),mat(numStates,numStates,fill::zeros));//now factor in the coupling type terms
cx_mat decayMatrix = cx_mat(mat(numStates,numStates,fill::zeros),mat(numStates,numStates,fill::zeros));//now factor in the coupling type terms


/* FUNCTIONS */ // you don't actually have to predeclare these, so there may be some "missing"
void init(double speed);                     // system initialization
void qstep(void);                    // advance quantum part of system
//void output(void);                   // output results
void outputData(void);                   // output results
int* getRabiComps(double x, double y, double z);

/***********************************************************************/
/*                                                                     */
/*  system initialization                                              */
/*                                                                     */
/***********************************************************************/

void init()
{
  int i;

  for(i=0;i<N0;i++)              // loop over particles in large box
    {
	
      //V[0][i] = 0;
      //V[1][i] = 0;
      //V[2][i] = 0.1;
      //V[0][i] = vSpread;
      //V[1][i] = 0;
      //V[2][i] = 0;
      //V[i] = velocityDistribution(rng);;
      double randPos0 = drand48();
      double randPos1 = drand48();
      double randPos2 = drand48();
      double randomTheta = drand48()*M_PI;
      double randomPhi = drand48()*2*M_PI;
      //V[i] = .02;
      //V[0][i] = 0;
      //V[1][i] = 0.1;
      //V[2][i] = 0;
      //V[0][i] = speed*sin(randomTheta)*cos(randomPhi);
      //V[1][i] = speed*sin(randomTheta)*sin(randomPhi);
      //V[2][i] = speed*cos(randomTheta);
      V[0][i] = velocityDistribution(rng);
      V[1][i] = velocityDistribution(rng);
      V[2][i] = velocityDistribution(rng);
      /*
      double roundVx = V[0][i]/(speed/10);
      V[0][i] = round(roundVx)*speed/10;
      double roundVy = V[1][i]/(speed/10);
      V[1][i] = round(roundVy)*speed/10;
      double roundVz = V[2][i]/(speed/10);
      V[2][i] = round(roundVz)*speed/10;
      */
      X[0][i] = 2*M_PI*randPos0;
      X[1][i] = 2*M_PI*randPos1;
      X[2][i] = 2*M_PI*randPos2;
      //X[0][i] = 0;
      //X[1][i] = 0;
      //X[2][i] =0;
      
      double rand1 = drand48();
      double rand2 = drand48();
      double rand3 = drand48();
      double rand4 = drand48();
      double rand5 = drand48();
      double rand6 = drand48();
      double rand7 = drand48();
      double rand8 = drand48();
      double rand9 = drand48();
      double rand10 = drand48();
      double rand11 = drand48();
      double rand12 = drand48();
      double wvFnOneComp = rand1/(rand1+rand2+rand3+rand4+rand5+rand6+rand7+rand8+rand9+rand10+rand11+rand12);
      double wvFnTwoComp = rand2/(rand1+rand2+rand3+rand4+rand5+rand6+rand7+rand8+rand9+rand10+rand11+rand12);
      double wvFnThreeComp = rand3/(rand1+rand2+rand3+rand4+rand5+rand6+rand7+rand8+rand9+rand10+rand11+rand12);
      double wvFnFourComp = rand4/(rand1+rand2+rand3+rand4+rand5+rand6+rand7+rand8+rand9+rand10+rand11+rand12);
      double wvFnFiveComp = rand5/(rand1+rand2+rand3+rand4+rand5+rand6+rand7+rand8+rand9+rand10+rand11+rand12);
      double wvFnSixComp = rand6/(rand1+rand2+rand3+rand4+rand5+rand6+rand7+rand8+rand9+rand10+rand11+rand12);
      double wvFnSevenComp = rand7/(rand1+rand2+rand3+rand4+rand5+rand6+rand7+rand8+rand9+rand10+rand11+rand12);
      double wvFnEightComp = rand8/(rand1+rand2+rand3+rand4+rand5+rand6+rand7+rand8+rand9+rand10+rand11+rand12);
      double wvFnNineComp = rand9/(rand1+rand2+rand3+rand4+rand5+rand6+rand7+rand8+rand9+rand10+rand11+rand12);
      double wvFnTenComp = rand10/(rand1+rand2+rand3+rand4+rand5+rand6+rand7+rand8+rand9+rand10+rand11+rand12);
      double wvFnElevenComp = rand11/(rand1+rand2+rand3+rand4+rand5+rand6+rand7+rand8+rand9+rand10+rand11+rand12);
      double wvFnTwelveComp = rand12/(rand1+rand2+rand3+rand4+rand5+rand6+rand7+rand8+rand9+rand10+rand11+rand12);

      double signRe2=1;
      double rand=drand48();
      if(rand<0.5){
	signRe2 =-1;
      }
      double signIm2=1;
      rand=drand48();
      if(rand<0.5){
	signIm2=-1;
      }

      double signRe3=1;
      rand=drand48();
      if(rand<0.5){
	signRe3 =-1;
      }
      double signIm3=1;
      rand=drand48();
      if(rand<0.5){
	signIm3=-1;
      }

      double signRe4=1;
      rand=drand48();
      if(rand<0.5){
	signRe4 =-1;
      }
      double signIm4=1;
      rand=drand48();
      if(rand<0.5){
	signIm4=-1;
      }

      double signRe5=1;
      rand=drand48();
      if(rand<0.5){
	signRe5 =-1;
      }
      double signIm5=1;
      rand=drand48();
      if(rand<0.5){
	signIm5=-1;
      }

      double signRe6=1;
      rand=drand48();
      if(rand<0.5){
	signRe6 =-1;
      }
      double signIm6=1;
      rand=drand48();
      if(rand<0.5){
	signIm6=-1;
      }

      double signRe7=1;
      rand=drand48();
      if(rand<0.5){
	signRe7 =-1;
      }
      double signIm7=1;
      rand=drand48();
      if(rand<0.5){
	signIm7=-1;
      }

      double signRe8=1;
      rand=drand48();
      if(rand<0.5){
	signRe8 =-1;
      }
      double signIm8=1;
      rand=drand48();
      if(rand<0.5){
	signIm8=-1;
      }

      double signRe9=1;
      rand=drand48();
      if(rand<0.5){
	signRe9 =-1;
      }
      double signIm9=1;
      rand=drand48();
      if(rand<0.5){
	signIm9=-1;
      }

      double signRe10=1;
      rand=drand48();
      if(rand<0.5){
	signRe10 =-1;
      }
      double signIm10=1;
      rand=drand48();
      if(rand<0.5){
	signIm10=-1;
      }

      double signRe11=1;
      rand=drand48();
      if(rand<0.5){
	signRe11 =-1;
      }
      double signIm11=1;
      rand=drand48();
      if(rand<0.5){
	signIm11=-1;
      }

      double signRe12=1;
      rand=drand48();
      if(rand<0.5){
	signRe12 =-1;
      }
      double signIm12=1;
      rand=drand48();
      if(rand<0.5){
	signIm12=-1;
      }

      
      
      double randRe2Cont = drand48();
      double randRe3Cont = drand48();
      double randRe4Cont = drand48();
      double randRe5Cont = drand48();
      double randRe6Cont = drand48();
      double randRe7Cont = drand48();
      double randRe8Cont = drand48();
      double randRe9Cont = drand48();
      double randRe10Cont = drand48();
      double randRe11Cont = drand48();
      double randRe12Cont = drand48();

      
      if(onlyFg1UpToFe1||onlyFg1UpToFe0){
	mat wvFn1Cont = sqrt(wvFnOneComp)*ident.col(4);
	mat wvFn2RealCont = signRe2*sqrt(wvFnTwoComp)*sqrt(randRe2Cont)*ident.col(5);
	mat wvFn2ImCont = signIm2*sqrt(wvFnTwoComp)*sqrt(1-randRe2Cont)*ident.col(5);
	mat wvFn3RealCont = signRe3*sqrt(wvFnThreeComp)*sqrt(randRe3Cont)*ident.col(6);
	mat wvFn3ImCont = signIm3*sqrt(wvFnThreeComp)*sqrt(1-randRe3Cont)*ident.col(6);
	wvFns[i]=cx_mat(wvFn1Cont+wvFn2RealCont+wvFn3RealCont,wvFn2ImCont+wvFn3ImCont);
	cx_mat wvFn = wvFns[i];
	double popS = std::norm(wvFn(0,0)) +std::norm(wvFn(1,0))+std::norm(wvFn(2,0))+std::norm(wvFn(3,0)) + std::norm(wvFn(4,0)) + std::norm(wvFn(5,0))+std::norm(wvFn(6,0)) + std::norm(wvFn(7,0)) + std::norm(wvFn(8,0)) + std::norm(wvFn(9,0))+ std::norm(wvFn(10,0))+ std::norm(wvFn(11,0));
	double popP = std::norm(wvFn(12,0))+std::norm(wvFn(13,0))+std::norm(wvFn(14,0))+std::norm(wvFn(15,0));
	wvFn = wvFn/sqrt(popS+popP);
	wvFns[i]=wvFn;
      }
      else if(onlyFg1UpToFe1||onlyFg1UpToFe0){
	mat wvFn1Cont = sqrt(wvFnOneComp)*ident.col(0);
	mat wvFn2RealCont = signRe2*sqrt(wvFnTwoComp)*sqrt(randRe2Cont)*ident.col(1);
	mat wvFn2ImCont = signIm2*sqrt(wvFnTwoComp)*sqrt(1-randRe2Cont)*ident.col(1);
	mat wvFn3RealCont = signRe3*sqrt(wvFnThreeComp)*sqrt(randRe3Cont)*ident.col(2);
	mat wvFn3ImCont = signIm3*sqrt(wvFnThreeComp)*sqrt(1-randRe3Cont)*ident.col(2);
	wvFns[i]=cx_mat(wvFn1Cont+wvFn2RealCont+wvFn3RealCont,wvFn2ImCont+wvFn3ImCont);
	cx_mat wvFn = wvFns[i];
	double popS = std::norm(wvFn(0,0)) +std::norm(wvFn(1,0))+std::norm(wvFn(2,0))+std::norm(wvFn(3,0)) + std::norm(wvFn(4,0)) + std::norm(wvFn(5,0))+std::norm(wvFn(6,0)) + std::norm(wvFn(7,0)) + std::norm(wvFn(8,0)) + std::norm(wvFn(9,0))+ std::norm(wvFn(10,0))+ std::norm(wvFn(11,0));
	double popP = std::norm(wvFn(12,0))+std::norm(wvFn(13,0))+std::norm(wvFn(14,0))+std::norm(wvFn(15,0));
	wvFn = wvFn/sqrt(popS+popP);
	wvFns[i]=wvFn;
	
      }
      else if(onlyFg0ToFe1){
	mat wvFn1Cont = ident.col(3);	
	mat wvFn2RealCont = 0*sqrt(wvFnTwoComp)*sqrt(randRe2Cont)*ident.col(5);
	mat wvFn2ImCont = 0*sqrt(wvFnTwoComp)*sqrt(1-randRe2Cont)*ident.col(5);
	mat wvFn3RealCont = 0*sqrt(wvFnThreeComp)*sqrt(randRe3Cont)*ident.col(6);
	mat wvFn3ImCont = 0*sqrt(wvFnThreeComp)*sqrt(1-randRe3Cont)*ident.col(6);
	wvFns[i]=cx_mat(wvFn1Cont+wvFn2RealCont+wvFn3RealCont,wvFn2ImCont+wvFn3ImCont);
	cx_mat wvFn = wvFns[i];
	double popS = std::norm(wvFn(0,0)) +std::norm(wvFn(1,0))+std::norm(wvFn(2,0))+std::norm(wvFn(3,0)) + std::norm(wvFn(4,0)) + std::norm(wvFn(5,0))+std::norm(wvFn(6,0)) + std::norm(wvFn(7,0)) + std::norm(wvFn(8,0)) + std::norm(wvFn(9,0))+ std::norm(wvFn(10,0))+ std::norm(wvFn(11,0));
	double popP = std::norm(wvFn(12,0))+std::norm(wvFn(13,0))+std::norm(wvFn(14,0))+std::norm(wvFn(15,0));
	wvFn = wvFn/sqrt(popS+popP);
	wvFns[i]=wvFn;
      }

      else if(onlyFg2ToFe1){
	mat wvFn1Cont = sqrt(wvFnOneComp)*ident.col(8);
	mat wvFn2RealCont = signRe2*sqrt(wvFnTwoComp)*sqrt(randRe2Cont)*ident.col(9);
	mat wvFn2ImCont = signIm2*sqrt(wvFnTwoComp)*sqrt(1-randRe2Cont)*ident.col(9);
	mat wvFn3RealCont = signRe3*sqrt(wvFnThreeComp)*sqrt(randRe3Cont)*ident.col(10);
	mat wvFn3ImCont = signIm3*sqrt(wvFnThreeComp)*sqrt(1-randRe3Cont)*ident.col(10);
	wvFns[i]=cx_mat(wvFn1Cont+wvFn2RealCont+wvFn3RealCont,wvFn2ImCont+wvFn3ImCont);
	cx_mat wvFn = wvFns[i];
	double popS = std::norm(wvFn(0,0)) +std::norm(wvFn(1,0))+std::norm(wvFn(2,0))+std::norm(wvFn(3,0)) + std::norm(wvFn(4,0)) + std::norm(wvFn(5,0))+std::norm(wvFn(6,0)) + std::norm(wvFn(7,0)) + std::norm(wvFn(8,0)) + std::norm(wvFn(9,0))+ std::norm(wvFn(10,0))+ std::norm(wvFn(11,0));
	double popP = std::norm(wvFn(12,0))+std::norm(wvFn(13,0))+std::norm(wvFn(14,0))+std::norm(wvFn(15,0));
	wvFn = wvFn/sqrt(popS+popP);
	wvFns[i]=wvFn;
      }

      else{
	mat wvFn1Cont = sqrt(wvFnOneComp)*ident.col(0);
	mat wvFn2RealCont = signRe2*sqrt(wvFnTwoComp)*sqrt(randRe2Cont)*ident.col(1);
	mat wvFn2ImCont = signIm2*sqrt(wvFnTwoComp)*sqrt(1-randRe2Cont)*ident.col(1);
	mat wvFn3RealCont = signRe3*sqrt(wvFnThreeComp)*sqrt(randRe3Cont)*ident.col(2);
	mat wvFn3ImCont = signIm3*sqrt(wvFnThreeComp)*sqrt(1-randRe3Cont)*ident.col(2);
	mat wvFn4RealCont = signRe4*sqrt(wvFnFourComp)*sqrt(randRe4Cont)*ident.col(3);
	mat wvFn4ImCont = signIm4*sqrt(wvFnFourComp)*sqrt(1-randRe4Cont)*ident.col(3);
	mat wvFn5RealCont = signRe5*sqrt(wvFnFiveComp)*sqrt(randRe5Cont)*ident.col(4);
	mat wvFn5ImCont = signIm5*sqrt(wvFnFiveComp)*sqrt(1-randRe5Cont)*ident.col(4);
	mat wvFn6RealCont = signRe6*sqrt(wvFnSixComp)*sqrt(randRe6Cont)*ident.col(5);
	mat wvFn6ImCont = signIm6*sqrt(wvFnSixComp)*sqrt(1-randRe6Cont)*ident.col(5);
	mat wvFn7RealCont = signRe7*sqrt(wvFnSevenComp)*sqrt(randRe7Cont)*ident.col(6);
	mat wvFn7ImCont = signIm7*sqrt(wvFnSevenComp)*sqrt(1-randRe7Cont)*ident.col(6);
	mat wvFn8RealCont = signRe8*sqrt(wvFnEightComp)*sqrt(randRe8Cont)*ident.col(7);
	mat wvFn8ImCont = signIm8*sqrt(wvFnEightComp)*sqrt(1-randRe8Cont)*ident.col(7);
	mat wvFn9RealCont = signRe9*sqrt(wvFnNineComp)*sqrt(randRe9Cont)*ident.col(8);
	mat wvFn9ImCont = signIm9*sqrt(wvFnNineComp)*sqrt(1-randRe9Cont)*ident.col(8);
	mat wvFn10RealCont = signRe10*sqrt(wvFnTenComp)*sqrt(randRe10Cont)*ident.col(9);
	mat wvFn10ImCont = signIm10*sqrt(wvFnTenComp)*sqrt(1-randRe10Cont)*ident.col(9);
	mat wvFn11RealCont = signRe11*sqrt(wvFnElevenComp)*sqrt(randRe11Cont)*ident.col(10);
	mat wvFn11ImCont = signIm11*sqrt(wvFnElevenComp)*sqrt(1-randRe11Cont)*ident.col(10);
	mat wvFn12RealCont = signRe12*sqrt(wvFnTwelveComp)*sqrt(randRe12Cont)*ident.col(11);
	mat wvFn12ImCont = signIm12*sqrt(wvFnTwelveComp)*sqrt(1-randRe12Cont)*ident.col(11);
	wvFns[i]=cx_mat(wvFn1Cont+wvFn2RealCont+wvFn3RealCont+wvFn4RealCont+wvFn5RealCont+wvFn6RealCont+wvFn7RealCont+wvFn8RealCont+wvFn9RealCont+wvFn10RealCont+wvFn11RealCont+wvFn12RealCont,wvFn2ImCont+wvFn3ImCont+wvFn4ImCont+wvFn5ImCont+wvFn6ImCont+wvFn7ImCont+wvFn8ImCont+wvFn9ImCont+wvFn10ImCont+wvFn11ImCont+wvFn12ImCont);
      }

      cx_mat wvFn = wvFns[i];
      if(removeFg1Down){
	
	wvFn(0,0).real(0);
	wvFn(0,0).imag(0);
	wvFn(1,0).real(0);
	wvFn(1,0).imag(0);
	wvFn(2,0).real(0);
	wvFn(2,0).imag(0);
      }

      if(removeFg0){
	wvFn(3,0).real(0);
	wvFn(3,0).imag(0);
      }

      if(removeFg1Up){
	wvFn(4,0).real(0);
	wvFn(4,0).imag(0);
	wvFn(5,0).real(0);
	wvFn(5,0).imag(0);
	wvFn(6,0).real(0);
	wvFn(6,0).imag(0);
      }

      if(removeFg2){
	wvFn(7,0).real(0);
	wvFn(7,0).imag(0);
	wvFn(8,0).real(0);
	wvFn(8,0).imag(0);
	wvFn(9,0).real(0);
	wvFn(9,0).imag(0);
	wvFn(10,0).real(0);
	wvFn(10,0).imag(0);
	wvFn(11,0).real(0);
	wvFn(11,0).imag(0);
      }
      double popS = std::norm(wvFn(0,0)) +std::norm(wvFn(1,0))+std::norm(wvFn(2,0))+std::norm(wvFn(3,0)) + std::norm(wvFn(4,0)) + std::norm(wvFn(5,0))+std::norm(wvFn(6,0)) + std::norm(wvFn(7,0)) + std::norm(wvFn(8,0)) + std::norm(wvFn(9,0))+ std::norm(wvFn(10,0))+ std::norm(wvFn(11,0));
      double popP = std::norm(wvFn(12,0))+std::norm(wvFn(13,0))+std::norm(wvFn(14,0))+std::norm(wvFn(15,0));
      wvFn = wvFn/sqrt(popS+popP);
      wvFns[i]=wvFn;

      tPart[i]=0.;
        
    }



}




/***********************************************************************/
/*                                                                     */
/*  advance quantum system                                               */
/*                                                                     */
/***********************************************************************/

void qstep(int count)
{  
  unsigned i,j,k;
  cx_mat wvFn,savedWvFn;
  //double tPart;
  mat zero_mat1=mat(1,1,fill::zeros);
  cx_mat dpmat,forceMatX,forceMatY,forceMatZ;
  double kickX,kickY,kickZ;
  cx_mat currTerm;
  //hamiltonian and various terms
  double dp,rand,dtHalf,prefactor;
  cx_mat hamCouplingTerm,dHamCouplingTermX,dHamCouplingTermY,dHamCouplingTermZ,hamWithoutDecay,hamil;
  cx_mat matPrefactor,wvFnStepped,k1,k2,k3,k4,wvFnk1,wvFnk2,wvFnk3;
  double rand2,rand3,randFExcState,randFGroundState;
  double norm13,norm14,norm15,norm16,totalNorm,totalFExcOneProb,totalFGroundTwoProb,totalFGroundOneUpProb,totalFGroundOneDownProb,prob13,prob14,prob15,prob16,randDir,z,y,x,probSigPlus,probPi,probSigMinus,randomX,randomY,randomZ;
  std::complex<double> sigPlusOmComp,dSigPlusOmCompX,dSigPlusOmCompY,dSigPlusOmCompZ;
  std::complex<double> sigMinusOmComp,dSigMinusOmCompX,dSigMinusOmCompY,dSigMinusOmCompZ;
  std::complex<double> piOmComp,dPiOmCompX,dPiOmCompY,dPiOmCompZ;
  std::complex<double> sigPlusOmComp2,dSigPlusOmCompX2,dSigPlusOmCompY2,dSigPlusOmCompZ2;
  std::complex<double> sigMinusOmComp2,dSigMinusOmCompX2,dSigMinusOmCompY2,dSigMinusOmCompZ2;
  std::complex<double> piOmComp2,dPiOmCompX2,dPiOmCompY2,dPiOmCompZ2;
  double popS,popP;
  double sumRealSigPlus;
  double sumImagSigPlus;
  double sumRealSigMinus;
  double sumImagSigMinus;
  double sumRealPi;
  double sumImagPi;
  

  /*begin parallel*/  //yeah i know it's a lot of variables...

  #pragma omp parallel private(i,j,k,wvFn,savedWvFn,zero_mat1,dpmat,kickX,kickY,kickZ,currTerm,dp,rand,dtHalf,prefactor,hamCouplingTerm,dHamCouplingTermX,dHamCouplingTermY,dHamCouplingTermZ,hamWithoutDecay,hamil,matPrefactor,wvFnStepped,k1,k2,k3,k4,wvFnk1,wvFnk2,wvFnk3,rand2,rand3,randFExcState,randFGroundState,norm13,norm14,norm15,norm16,totalNorm,totalFExcOneProb,totalFGroundTwoProb,totalFGroundOneUpProb,totalFGroundOneDownProb,prob13,prob14,prob15,prob16,randDir,x,y,z,probSigPlus,probPi,probSigMinus,randomX,randomY,randomZ,sigPlusOmComp,dSigPlusOmCompX,dSigPlusOmCompY,dSigPlusOmCompZ,sigMinusOmComp,dSigMinusOmCompX,dSigMinusOmCompY,dSigMinusOmCompZ,piOmComp,dPiOmCompX,dPiOmCompY,dPiOmCompZ,sigPlusOmComp2,dSigPlusOmCompX2,dSigPlusOmCompY2,dSigPlusOmCompZ2,sigMinusOmComp2,dSigMinusOmCompX2,dSigMinusOmCompY2,dSigMinusOmCompZ2,piOmComp2,dPiOmCompX2,dPiOmCompY2,dPiOmCompZ2,popS,popP,sumRealSigPlus,sumImagSigPlus,sumRealSigMinus,sumImagSigMinus,sumRealPi,sumImagPi,forceMatX,forceMatY,forceMatZ) shared(V,X,wvFns,cs,wvFn1,wvFn2,wvFn3,wvFn4,wvFn5,wvFn6,wvFn7,wvFn8,wvFn9,wvFn10,wvFn11,wvFn12,wvFn13,wvFn14,wvFn15,wvFn16,hamCouplingTermOnlySigPlus,hamCouplingTermOnlySigMinus,hamCouplingTermOnlyPi,hamEnergyTerm,hamDecayTerm,decayMatrix,vKick,gs,I,numStates,tPart,Om,currFluorCounts)
  {
  #pragma omp for 
 
  
  for(i=0;i<N0;i++){//for every wavefunction: evolve it
  //get rabi comps
  //z=X[i];
  //sigPlusOmComp = -sin(z)-I*cos(z);
  //sigMinusOmComp = sin(z)-I*cos(z);
  //sigPlusOmComp = I*sin(z)+cos(z);
  //sigMinusOmComp = -I*sin(z)+cos(z);
  //sigPlusOmComp = -1*(1+I)/sqrt(2)*(cos(z)+sin(z));
  //sigPlusOmComp = -1/sqrt(2)*(cos(z)-sin(z)+I*cos(z)-I*sin(z));
  //sigMinusOmComp = (1-I)/sqrt(2)*(cos(z)-sin(z));
  //sigMinusOmComp = 1/sqrt(2)*(cos(z)+sin(z)-I*cos(z)-I*sin(z));
  //piOmComp = I*2.0*sin(y);

      
  dpmat=cx_mat(zero_mat1,zero_mat1);
  wvFn=wvFns[i];
  //velQuant=V[i];//use x velocity
  x=X[0][i];
  y=X[1][i];
  z=X[2][i];
  tPart[i] += dt;
  //for(j=0;j<36;j++){
  //dpmatTerms[j] = dt*wvFn.t()*cs[j].t()*cs[j]*wvFn;
    //dpmatTerms[j].print("dpTerms:");
    //dpmat.print("dpMat:");
    //zero_mat1.print("zeroMat:");
    //dpmat=dpmat+dpmatTerms[j]*gs[j]*gs[j];
      
  //}//for all states, calculate dpmat
  dpmat = dt*wvFn.t()*decayMatrix*wvFn;
  dp = dpmat(0,0).real();
  rand = drand48();
  
  if(rand>dp)//if no jump, evolve according to non-Hermitian Hamiltonian (see Lukin book or TKL PhD Thesis)
    {
      
      sumRealSigPlus=0;
      sumImagSigPlus=0;
      
      sigPlusOmComp = I*(-sin(z)+cos(y))+(cos(z)+sin(x));
      sigMinusOmComp = -I*(-sin(z)+cos(y))+(cos(z)+sin(x));
      piOmComp = -I*sqrt(2)*(sin(y)+cos(x));
      
      dSigPlusOmCompX = cos(x)+0.0*I;
      dSigMinusOmCompX = cos(x)+0.0*I;
      dPiOmCompX = I*sqrt(2)*sin(x);
      
      dSigPlusOmCompY = -1.0*I*sin(y);
      dSigMinusOmCompY = I*sin(y);
      dPiOmCompY = -I*sqrt(2)*cos(y);
      
      dSigPlusOmCompZ = -I*cos(z)-sin(z);
      dSigMinusOmCompZ = I*cos(z)-sin(z);
      dPiOmCompZ = 0;
      //unclear to me if this next bit is needed; this not-so-random phase accrues from the fact that the wavevector for each 'laser' is different
      //(\delta(k) = 2*pi*(frequency_Difference)/c * L), where L is path length from initial and second pass
      //Here I've just put some reasonable values (this is what you get assuming 19.6\Gamma and 1 foot total 'extra' path length between initial and final pass)
      //This is obviously true if you have two different lasers, but I don't know if you have to do this when producing sideband via phase modulation?
      //There may also be some kind of need to take a 'random' phase from laser jitter?  Not sure exactly...
      
      double randPhaseX = detLaserDiff*gamOverC*retroPathLength;
      double randPhaseY = detLaserDiff*gamOverC*retroPathLength;
      double randPhaseZ = detLaserDiff*gamOverC*retroPathLength;
      
      sigPlusOmComp2 = I*(-sin(z+randPhaseZ)+cos(y+randPhaseY))+(cos(z+randPhaseZ)+sin(x+randPhaseX));
      sigMinusOmComp2 = -I*(-sin(z+randPhaseZ)+cos(y+randPhaseY))+(cos(z+randPhaseZ)+sin(x+randPhaseX));
      piOmComp2 = -I*sqrt(2)*(sin(y+randPhaseY)+cos(x+randPhaseX));
      
      dSigPlusOmCompX2 = cos(x+randPhaseX)+0.0*I;
      dSigMinusOmCompX2 = cos(x+randPhaseX)+0.0*I;
      dPiOmCompX2 = I*sqrt(2)*sin(x+randPhaseX);
      
      dSigPlusOmCompY2 = -1.0*I*sin(y+randPhaseY);
      dSigMinusOmCompY2 = I*sin(y+randPhaseY);
      dPiOmCompY2 = -I*sqrt(2)*cos(y+randPhaseY);
      
      dSigPlusOmCompZ2 = -I*cos(z+randPhaseZ)-sin(z+randPhaseZ);
      dSigMinusOmCompZ2 = I*cos(z+randPhaseZ)-sin(z+randPhaseZ);
      dPiOmCompZ2 = 0;
      

 
      hamCouplingTerm = (Om*sigPlusOmComp+OmRepump*sigPlusOmComp2*exp(I*detLaserDiff*tPart[i]))/2.0*hamCouplingTermOnlySigPlus+(Om*sigMinusOmComp+OmRepump*sigMinusOmComp2*exp(I*detLaserDiff*tPart[i]))/2.0*hamCouplingTermOnlySigMinus+(Om*piOmComp+OmRepump*piOmComp2*exp(I*detLaserDiff*tPart[i]))/2.0*hamCouplingTermOnlyPi;
      
      dHamCouplingTermX = (Om*dSigPlusOmCompX+OmRepump*dSigPlusOmCompX2*exp(I*detLaserDiff*tPart[i]))/2.0*hamCouplingTermOnlySigPlus+(Om*dSigMinusOmCompX+OmRepump*dSigMinusOmCompX2*exp(I*detLaserDiff*tPart[i]))/2.0*hamCouplingTermOnlySigMinus+(Om*dPiOmCompX+OmRepump*dPiOmCompX2*exp(I*detLaserDiff*tPart[i]))/2.0*hamCouplingTermOnlyPi;
      
      dHamCouplingTermY = (Om*dSigPlusOmCompY+OmRepump*dSigPlusOmCompY2*exp(I*detLaserDiff*tPart[i]))/2.0*hamCouplingTermOnlySigPlus+(Om*dSigMinusOmCompY+OmRepump*dSigMinusOmCompY2*exp(I*detLaserDiff*tPart[i]))/2.0*hamCouplingTermOnlySigMinus+(Om*dPiOmCompY+OmRepump*dPiOmCompY2*exp(I*detLaserDiff*tPart[i]))/2.0*hamCouplingTermOnlyPi;
      
      dHamCouplingTermZ = (Om*dSigPlusOmCompZ+OmRepump*dSigPlusOmCompZ2*exp(I*detLaserDiff*tPart[i]))/2.0*hamCouplingTermOnlySigPlus+(Om*dSigMinusOmCompZ+OmRepump*dSigMinusOmCompZ2*exp(I*detLaserDiff*tPart[i]))/2.0*hamCouplingTermOnlySigMinus+(Om*dPiOmCompZ+OmRepump*dPiOmCompZ2*exp(I*detLaserDiff*tPart[i]))/2.0*hamCouplingTermOnlyPi;
      

      
      forceMatX = -1*vKick*dt*wvFn.t()*(dHamCouplingTermX+dHamCouplingTermX.t())*wvFn;
      forceMatY = -1*vKick*dt*wvFn.t()*(dHamCouplingTermY+dHamCouplingTermY.t())*wvFn;
      forceMatZ = -1*vKick*dt*wvFn.t()*(dHamCouplingTermZ+dHamCouplingTermZ.t())*wvFn;
      kickX = forceMatX(0,0).real();
      kickY = forceMatY(0,0).real();
      kickZ = forceMatZ(0,0).real();

      
      hamWithoutDecay=hamEnergyTerm+hamCouplingTerm+hamCouplingTerm.t();//add all the non-decay terms together, including hermitian conjugates of coupling terms!
      	
      hamil=hamWithoutDecay+hamDecayTerm;//total Hamiltonian for non-hermitian evolution
	
      //with hamiltonian calculated, can evolve wvFn using RK method (I choose 3/8 method)

      dtHalf = dt/2;
      matPrefactor = ident-I*dt*hamil;
      dpmat=cx_mat(zero_mat1,zero_mat1);
      
      //get k1,y1 (k1 is slope at t0 calculated using y0, y0 is initial wvFn value.  y1 (wvFnk1) is wvFn stepped by dt/2 w/ slope k1)

      dpmat = dt*wvFn.t()*decayMatrix*wvFn;
      dp = dpmat(0,0).real();
      prefactor = 1/sqrt(1-dp);
      
      wvFnStepped = prefactor*matPrefactor*wvFn;
      k1 = 1./(dt)*(wvFnStepped-wvFn);
      wvFnk1 = wvFn+dtHalf*k1;
	
      //get k2,y2 (k2 is slope at t0+dt/2 calculated using y1, y2 (wvFnk2) is wvFn stepped by dt/2 w/ slope k2)
     
      dpmat = dt*wvFnk1.t()*decayMatrix*wvFnk1;
      dp = dpmat(0,0).real();
      prefactor = 1/sqrt(1-dp);
      
      wvFnStepped = prefactor*matPrefactor*wvFnk1;
      k2 = 1./(dt)*(wvFnStepped-wvFnk1);
      wvFnk2 = wvFn+dtHalf*k2;

	
      //get k3, y3 (k3 is slope at t0+dt/2 calculated using y2, y3 (wvFnk3) is wvFn stepped by dt w/ slope k3)
	
    
      dpmat = dt*wvFnk2.t()*decayMatrix*wvFnk2;
      dp = dpmat(0,0).real();
      prefactor = 1/sqrt(1-dp);
      
      wvFnStepped = prefactor*matPrefactor*wvFnk2;
      k3 = 1./(dt)*(wvFnStepped-wvFnk2);
      wvFnk3 = wvFn+dt*k3;
	
      //get k4, yfinal (k4 is slope at t0+dt calculated using y3, yfinal is wvFn stepped by dt using weighted average of k1,k2,k3, and k4)

      dpmat = dt*wvFnk3.t()*decayMatrix*wvFnk3;
      dp = dpmat(0,0).real();
      prefactor = 1/sqrt(1-dp);
      
      wvFnStepped = prefactor*matPrefactor*wvFnk3;
      k4 = 1./(dt)*(wvFnStepped-wvFnk3);
      
      wvFn = wvFn+(k1+3*k2+3*k3+k4)/8*(dt);//finally: evolve the wavefunction according to completion of runge-kutta propagator
 
    }
   else{//else if there was a "jump" roll again for polarization and then which state is jumped to
     //I've seen this done in different ways in the literature.  Here's my approach: In principle, if not in practice, the photon energy is knowable
     //(imagine a sphere comprised of a very fine line-filter surrounding the apparatus, for example)
     //Thus, you should roll for which state you decayed from and which state you decayed into, instead of being 'agnostic' about it in which case you would
     //simply apply the 'coupling' operator to the excited state fraction of the pre-jump wavefunction
     //But, since there is no B-Field (here at least), the zeeman levels within a hyperfine manifold are degenerate, and so once an excited state hyperfine manifold
     //and ground state manifold are 'rolled' for for the randomly selected photon polarization, the new ground state has populations in each allowable Zeeman state for the
     //given polarization, with weight based on the excited state Zeeman populations of the 'pre-jump' wavefunction

     //e.g., for a re-normalized excited state of 1/2|F'=1,m'=1>-1/2|F'=1,m'=0>+1/sqrt(2)|F'=0,m'=0>, we first roll for polarization.  Let's assume it's \sigma^-.  We then
     //roll for which excited state manifold the decay, with weights from the excited state wavefunction conditioned on the chosen polarization.  Let's assume that turned out to be |F'=1>.
     //Finally, we roll for what
     //ground state it decays to, let's say the |F=1,J=1/2> manifold.  Then, the jumped wavefunction winds up becoming  1/sqrt(2)|F=1,J=1/2,m=0> - 1/sqrt(2)|F=1,J=1/2,m=-1>
     //where the signs reflect the pre-jump wavefunction signs.
     
    //wvFn.print("WvBefore:");
    //cout<<"jump\n";
    //kick=0;
    currFluorCounts++;
    randomX = 2*drand48()-1;
    randomY = 2*drand48()-1;
    randomZ = 2*drand48()-1;
    kickX = vKick*randomX/(randomX*randomX+randomY*randomY+randomZ*randomZ);
    kickY = vKick*randomY/(randomX*randomX+randomY*randomY+randomZ*randomZ);
    kickZ = vKick*randomZ/(randomX*randomX+randomY*randomY+randomZ*randomZ);  
    
    randomX = 2*drand48()-1;
    randomY = 2*drand48()-1;
    randomZ = 2*drand48()-1;
    kickX += vKick*randomX/(randomX*randomX+randomY*randomY+randomZ*randomZ);
    kickY += vKick*randomY/(randomX*randomX+randomY*randomY+randomZ*randomZ);
    kickZ += vKick*randomZ/(randomX*randomX+randomY*randomY+randomZ*randomZ);//do it twice
    
    rand2 = drand48();
    randFExcState = drand48();
    randFGroundState = drand48();
    norm13 = std::norm(wvFn(12,0));
    norm14 = std::norm(wvFn(13,0));
    norm15 = std::norm(wvFn(14,0));
    norm16 = std::norm(wvFn(15,0));
    totalNorm = norm13+norm14+norm15+norm16;
    //totalFThreeHalvesNorm = norm7+norm8+norm9+norm10;
    //cout<<norm6<<"\n";
    //cout<<totalNorm<<"\n";
    prob13=norm13/totalNorm;
    prob14=norm14/totalNorm;
    prob15=norm15/totalNorm;
    prob16=norm16/totalNorm;
    //totalFExcThreeHalvesProb = prob7+prob8+prob9+prob10;
    
    probSigPlus = (gs[0]*gs[0]+gs[4]*gs[4]+gs[8]*gs[8])*prob14+(gs[2]*gs[2]+gs[3]*gs[3]+gs[6]*gs[6]+gs[10]*gs[10])*prob13+gs[7]*gs[7]*prob15+(gs[1]*gs[1]+gs[5]*gs[5]+gs[9]*gs[9])*prob16;
    
    probPi = (gs[23]*gs[23]+gs[26]*gs[26]+gs[29]*gs[29]+gs[33]*gs[33])*prob14+(gs[22]*gs[22]+gs[28]*gs[28]+gs[32]*gs[32])*prob15+(gs[25]*gs[25]+gs[31]*gs[31]+gs[35]*gs[35])*prob13+(gs[24]*gs[24]+gs[27]*gs[27]+gs[30]*gs[30]+gs[34]*gs[34])*prob16;
    
    probSigMinus = (gs[12]*gs[12]+gs[16]*gs[16]+gs[19]*gs[19])*prob14+(gs[11]*gs[11]+gs[14]*gs[14]+gs[15]*gs[15]+gs[18]*gs[18])*prob15+gs[21]*gs[21]*prob13+(gs[13]*gs[13]+gs[17]*gs[17]+gs[20]*gs[20])*prob16;
    //wvFn.zeros();
    tPart[i]=0;
    randDir = drand48();
    //decaysS=decaysS+1;
    //kick = vKick*cos(randDir*M_PI);
    savedWvFn = wvFn; 
      
    if (rand2<probSigPlus)//sig plus emit
      {
	
	//cout<<"sigPlus\n";
	wvFn.zeros();
        totalFExcOneProb = ((gs[0]*gs[0]+gs[4]*gs[4]+gs[8]*gs[8])*prob14+(gs[2]*gs[2]+gs[3]*gs[3]+gs[6]*gs[6]+gs[10]*gs[10])*prob13+gs[7]*gs[7]*prob15)/probSigPlus;
	if(randFExcState<totalFExcOneProb){
	  //totalFGroundThreeHalvesProb =
	  //totalFGroundTwoProb = (gs[1]*gs[1]*prob7+gs[4]*gs[4]*prob8+gs[7]*gs[7]*prob9)/totalFExcOneProb;
	  totalFGroundTwoProb = (gs[8]*gs[8]*prob14+gs[10]*gs[10]*prob13+gs[7]*gs[7]*prob15)/totalFExcOneProb/probSigPlus;
	  totalFGroundOneUpProb = (gs[4]*gs[4]*prob14+gs[6]*gs[6]*prob13)/totalFExcOneProb/probSigPlus;
	  totalFGroundOneDownProb = (gs[0]*gs[0]*prob14+gs[2]*gs[2]*prob13)/totalFExcOneProb/probSigPlus;
	    
	  if(randFGroundState<totalFGroundTwoProb){
	    wvFn(8,0).real(savedWvFn(13,0).real()*gs[8]);
	    wvFn(8,0).imag(savedWvFn(13,0).imag()*gs[8]);
	    wvFn(9,0).real(savedWvFn(12,0).real()*gs[10]);
	    wvFn(9,0).imag(savedWvFn(12,0).imag()*gs[10]);
	    wvFn(7,0).real(savedWvFn(14,0).real()*gs[7]);
	    wvFn(7,0).imag(savedWvFn(14,0).imag()*gs[7]);
	  }
	  
	  else if(randFGroundState<totalFGroundTwoProb+totalFGroundOneUpProb){
	    wvFn(4,0).real(savedWvFn(13,0).real()*gs[4]);
	    wvFn(4,0).imag(savedWvFn(13,0).imag()*gs[4]);
	    wvFn(5,0).real(savedWvFn(12,0).real()*gs[6]);
	    wvFn(5,0).imag(savedWvFn(12,0).imag()*gs[6]);
	  }

	  else if(randFGroundState<totalFGroundTwoProb+totalFGroundOneUpProb+totalFGroundOneDownProb){
	    wvFn(0,0).real(savedWvFn(13,0).real()*gs[0]);
	    wvFn(0,0).imag(savedWvFn(13,0).imag()*gs[0]);
	    wvFn(1,0).real(savedWvFn(12,0).real()*gs[2]);
	    wvFn(1,0).imag(savedWvFn(12,0).imag()*gs[2]);
	  }
	  else{
	    wvFn(3,0).real(savedWvFn(12,0).real()*gs[3]);
	    wvFn(3,0).imag(savedWvFn(12,0).imag()*gs[3]);

	  }
	  
	}

	else{
	  
	  totalFGroundTwoProb = (gs[9]*gs[9]*prob16)/(1-totalFExcOneProb)/probSigPlus;
	  totalFGroundOneUpProb = (gs[5]*gs[5]*prob16)/(1-totalFExcOneProb)/probSigPlus;
	  totalFGroundOneDownProb = (gs[1]*gs[1]*prob16)/(1-totalFExcOneProb)/probSigPlus;
	  
	  if(randFGroundState<totalFGroundTwoProb){
	    wvFn(8,0).real(savedWvFn(15,0).real()*gs[9]);
	    wvFn(8,0).imag(savedWvFn(15,0).imag()*gs[9]);
	  }
	  
	  else if(randFGroundState<totalFGroundTwoProb+totalFGroundOneUpProb){
	    wvFn(4,0).real(savedWvFn(15,0).real()*gs[5]);
	    wvFn(4,0).imag(savedWvFn(15,0).imag()*gs[5]);
	  }

	  else if(randFGroundState<totalFGroundTwoProb+totalFGroundOneUpProb+totalFGroundOneDownProb){
	    wvFn(0,0).real(savedWvFn(15,0).real()*gs[1]);
	    wvFn(0,0).imag(savedWvFn(15,0).imag()*gs[1]);
	  }
	  
	  else{
	    wvFn(0,0).real(savedWvFn(15,0).real()*gs[1]);
	    wvFn(0,0).imag(savedWvFn(15,0).imag()*gs[1]);

	  }
	  
	}
	
      }
   
    else if(rand2<probSigPlus+probPi)//pi emit
      {
	//cout<<"Pi\n";
	wvFn.zeros();
        totalFExcOneProb = ((gs[23]*gs[23]+gs[26]*gs[26]+gs[29]*gs[29]+gs[33]*gs[33])*prob14+(gs[22]*gs[22]+gs[28]*gs[28]+gs[32]*gs[32])*prob15+(gs[25]*gs[25]+gs[31]*gs[31]+gs[35]*gs[35])*prob13)/probPi;
	if(randFExcState<totalFExcOneProb){
	  //totalFGroundThreeHalvesProb =
	  //totalFGroundTwoProb = (gs[1]*gs[1]*prob7+gs[4]*gs[4]*prob8+gs[7]*gs[7]*prob9)/totalFExcOneProb;
	  totalFGroundTwoProb = (gs[33]*gs[33]*prob14+gs[35]*gs[35]*prob13+gs[32]*gs[32]*prob15)/totalFExcOneProb/probPi;
	  totalFGroundOneUpProb = (gs[29]*gs[29]*prob14+gs[31]*gs[31]*prob13+gs[28]*gs[28]*prob15)/totalFExcOneProb/probPi;
	  totalFGroundOneDownProb = (gs[23]*gs[23]*prob14+gs[25]*gs[25]*prob13+gs[22]*gs[22]*prob15)/totalFExcOneProb/probPi;
	  
	  if(randFGroundState<totalFGroundTwoProb){
	    //cout<<"A\n";
	    wvFn(9,0).real(savedWvFn(13,0).real()*gs[33]);
	    wvFn(9,0).imag(savedWvFn(13,0).imag()*gs[33]);
	    wvFn(10,0).real(savedWvFn(12,0).real()*gs[35]);
	    wvFn(10,0).imag(savedWvFn(12,0).imag()*gs[35]);
	    wvFn(8,0).real(savedWvFn(14,0).real()*gs[32]);
	    wvFn(8,0).imag(savedWvFn(14,0).imag()*gs[32]);
	  }
	  
	  else if(randFGroundState<totalFGroundTwoProb+totalFGroundOneUpProb){
	    //cout<<"B\n";
	    wvFn(5,0).real(savedWvFn(13,0).real()*gs[29]);
	    wvFn(5,0).imag(savedWvFn(13,0).imag()*gs[29]);
	    wvFn(6,0).real(savedWvFn(12,0).real()*gs[31]);
	    wvFn(6,0).imag(savedWvFn(12,0).imag()*gs[31]);
	    wvFn(4,0).real(savedWvFn(14,0).real()*gs[28]);
	    wvFn(4,0).imag(savedWvFn(14,0).imag()*gs[28]);
	  }

	  else if(randFGroundState<totalFGroundTwoProb+totalFGroundOneUpProb+totalFGroundOneDownProb){
	    //cout<<"C\n";
	    wvFn(1,0).real(savedWvFn(13,0).real()*gs[23]);
	    wvFn(1,0).imag(savedWvFn(13,0).imag()*gs[23]);
	    wvFn(2,0).real(savedWvFn(12,0).real()*gs[25]);
	    wvFn(2,0).imag(savedWvFn(12,0).imag()*gs[25]);
	    wvFn(0,0).real(savedWvFn(14,0).real()*gs[22]);
	    wvFn(0,0).imag(savedWvFn(14,0).imag()*gs[22]);
	  }
	  else{
	    //cout<<"D\n";
	    wvFn(3,0).real(savedWvFn(13,0).real()*gs[26]);
	    wvFn(3,0).imag(savedWvFn(13,0).imag()*gs[26]);
	    
	  }
	  
	}
	
	else{
	  
	  totalFGroundTwoProb = (gs[34]*gs[34]*prob16)/(1-totalFExcOneProb)/probPi;
	  totalFGroundOneUpProb = (gs[30]*gs[30]*prob16)/(1-totalFExcOneProb)/probPi;
	  totalFGroundOneDownProb = (gs[24]*gs[24]*prob16)/(1-totalFExcOneProb)/probPi;
	  //cout<<totalFGroundTwoProb<<"\n";
	  //cout<<totalFGroundOneUpProb<<"\n";
	  //cout<<totalFGroundOneDownProb<<"\n";
	  
	  if(randFGroundState<totalFGroundTwoProb){
	    //cout<<"E\n";
	    wvFn(9,0).real(savedWvFn(15,0).real()*gs[34]);
	    wvFn(9,0).imag(savedWvFn(15,0).imag()*gs[34]);
	  }
	  
	  else if(randFGroundState<totalFGroundTwoProb+totalFGroundOneUpProb){
	    //cout<<"F\n";
	    wvFn(5,0).real(savedWvFn(15,0).real()*gs[30]);
	    wvFn(5,0).imag(savedWvFn(15,0).imag()*gs[30]);
	  }

	  else if(randFGroundState<totalFGroundTwoProb+totalFGroundOneUpProb+totalFGroundOneDownProb){
	    //cout<<"G\n";
	    wvFn(1,0).real(savedWvFn(15,0).real()*gs[24]);
	    wvFn(1,0).imag(savedWvFn(15,0).imag()*gs[24]);
	  }
	  
	  else{
	    wvFn(3,0).real(savedWvFn(15,0).real()*gs[27]);
	    //cout<<"H\n";
	    wvFn(3,0).imag(savedWvFn(15,0).imag()*gs[27]);

	  }
	  
	}
	
      }
  
	  
  
    else//sig minus emit
      {
	//cout<<"sigMin\n";
	wvFn.zeros();
        totalFExcOneProb = ((gs[12]*gs[12]+gs[16]*gs[16]+gs[19]*gs[19])*prob14+(gs[11]*gs[11]+gs[14]*gs[14]+gs[15]*gs[15]+gs[18]*gs[18])*prob15+gs[21]*gs[21]*prob13)/probSigMinus;
	
	if(randFExcState<totalFExcOneProb){
	  //totalFGroundThreeHalvesProb =
	  //totalFGroundTwoProb = (gs[1]*gs[1]*prob7+gs[4]*gs[4]*prob8+gs[7]*gs[7]*prob9)/totalFExcOneProb;
	  totalFGroundTwoProb = (gs[19]*gs[19]*prob14+gs[21]*gs[21]*prob13+gs[18]*gs[18]*prob15)/totalFExcOneProb/probSigMinus;
	  totalFGroundOneUpProb = (gs[16]*gs[16]*prob14+gs[15]*gs[15]*prob15)/totalFExcOneProb/probSigMinus;
	  totalFGroundOneDownProb = (gs[12]*gs[12]*prob14+gs[11]*gs[11]*prob15)/totalFExcOneProb/probSigMinus;
	    
	  if(randFGroundState<totalFGroundTwoProb){
	    wvFn(10,0).real(savedWvFn(13,0).real()*gs[19]);
	    wvFn(10,0).imag(savedWvFn(13,0).imag()*gs[19]);
	    wvFn(11,0).real(savedWvFn(12,0).real()*gs[21]);
	    wvFn(11,0).imag(savedWvFn(12,0).imag()*gs[21]);
	    wvFn(9,0).real(savedWvFn(14,0).real()*gs[18]);
	    wvFn(9,0).imag(savedWvFn(14,0).imag()*gs[18]);
	  }
	  
	  else if(randFGroundState<totalFGroundTwoProb+totalFGroundOneUpProb){
	    wvFn(6,0).real(savedWvFn(13,0).real()*gs[16]);
	    wvFn(6,0).imag(savedWvFn(13,0).imag()*gs[16]);
	    wvFn(5,0).real(savedWvFn(14,0).real()*gs[15]);
	    wvFn(5,0).imag(savedWvFn(14,0).imag()*gs[15]);
	  }

	  else if(randFGroundState<totalFGroundTwoProb+totalFGroundOneUpProb+totalFGroundOneDownProb){
	    wvFn(2,0).real(savedWvFn(13,0).real()*gs[12]);
	    wvFn(2,0).imag(savedWvFn(13,0).imag()*gs[12]);
	    wvFn(1,0).real(savedWvFn(14,0).real()*gs[11]);
	    wvFn(1,0).imag(savedWvFn(14,0).imag()*gs[11]);
	  }
	  else{
	    wvFn(3,0).real(savedWvFn(14,0).real()*gs[14]);
	    wvFn(3,0).imag(savedWvFn(14,0).imag()*gs[14]);

	  }
	  
	}

	else{
	  
	  totalFGroundTwoProb = (gs[20]*gs[20]*prob16)/(1-totalFExcOneProb)/probSigMinus;
	  totalFGroundOneUpProb = (gs[17]*gs[17]*prob16)/(1-totalFExcOneProb)/probSigMinus;
	  totalFGroundOneDownProb = (gs[13]*gs[13]*prob16)/(1-totalFExcOneProb)/probSigMinus;
	  
	  if(randFGroundState<totalFGroundTwoProb){
	    wvFn(10,0).real(savedWvFn(15,0).real()*gs[20]);
	    wvFn(10,0).imag(savedWvFn(15,0).imag()*gs[20]);
	  }
	  
	  else if(randFGroundState<totalFGroundTwoProb+totalFGroundOneUpProb){
	    wvFn(6,0).real(savedWvFn(15,0).real()*gs[17]);
	    wvFn(6,0).imag(savedWvFn(15,0).imag()*gs[17]);
	  }

	  else if(randFGroundState<totalFGroundTwoProb+totalFGroundOneUpProb+totalFGroundOneDownProb){
	    wvFn(2,0).real(savedWvFn(15,0).real()*gs[13]);
	    wvFn(2,0).imag(savedWvFn(15,0).imag()*gs[13]);
	  }
	  
	  else{
	    wvFn(2,0).real(savedWvFn(15,0).real()*gs[13]);
	    wvFn(2,0).imag(savedWvFn(15,0).imag()*gs[13]);

	  }
	  
	}
      }
    
    if(!applySpontForce){
      kickX=0;
      kickY=0;
      kickZ=0;
    }
    //cout<<kick<<"\n";
   }
 
  if(applyForce){
    V[0][i]=V[0][i]+kickX;
    V[1][i]=V[1][i]+kickY;
    V[2][i]=V[2][i]+kickZ;
  }
  
  wvFns[i] = wvFn;

  if(reNormalizewvFns){
    popS = std::norm(wvFn(0,0)) +std::norm(wvFn(1,0))+std::norm(wvFn(2,0))+std::norm(wvFn(3,0)) + std::norm(wvFn(4,0)) + std::norm(wvFn(5,0))+std::norm(wvFn(6,0)) + std::norm(wvFn(7,0)) + std::norm(wvFn(8,0)) + std::norm(wvFn(9,0))+ std::norm(wvFn(10,0))+ std::norm(wvFn(11,0));
    popP = std::norm(wvFn(12,0))+std::norm(wvFn(13,0))+std::norm(wvFn(14,0))+std::norm(wvFn(15,0));
    wvFn = wvFn/sqrt(popS+popP);
    wvFns[i]=wvFn;
  }

  }//for all particles, evolve wave function
  }//pragmaOmp
}


/*******************************************************************/
/*******************************************************************/
/*************************STEP**************************************/
/*******************************************************************/
/*******************************************************************/

void step(void){
  for (int i=0;i<N0;i++){
    X[0][i] += TIMESTEP*V[0][i];
    X[1][i] += TIMESTEP*V[1][i];
    X[2][i] += TIMESTEP*V[2][i];
  }
}



void outputData(int counter)
{
  unsigned i,j;
  double V2;         // width of gaussian weight function for Pvel(vel)
  double velXAvg=0.0;
  //double EkinX,EkinY,EkinZ;       // average kinetic energy along each direction
  FILE *fa;
  FILE *faX;
  FILE *faY;
  FILE *faZ;
  FILE *fa2;
  FILE *fa3;
  char dataDirCopy[256];
  char buffer[256];
  char fileName[256];
  
  char dataDirCopyX[256];
  char bufferX[256];
  char fileNameX[256];
  
  char dataDirCopyY[256];
  char bufferY[256];
  char fileNameY[256];
  
  char dataDirCopyZ[256];
  char bufferZ[256];
  char fileNameZ[256];

  cx_mat densMatrix,p37,p28,p19,p58,p67,p48,p39,p210,p69,p510,p110,p29,p38,p47,p59,p68,currTerm;

  
  double forceX=0,forceY=0,forceZ=0,FDotVNorm=0,meanVelSq=0;
  double popFg2mg2=0,popFg2mg1=0,popFg2mg0=0,popFg2mgMin1=0,popFg2mgMin2=0,popFg1Upmg1=0,popFg1Upmg0=0,popFg1UpmgMin1=0,popFg1Downmg1=0,popFg1Downmg0=0,popFg1DownmgMin1=0,popFg0mg0=0,popFe1me1=0,popFe1me0=0,popFe1meMin1=0,popFe0me0=0;
  double vxDist = 0, vyDist = 0,vzDist = 0;
  cx_mat currWvFn;
  
  strcpy(dataDirCopy,saveDirectory);
  snprintf(buffer, 256,"statePopulationsVsVTime.dat");
  strcpy(fileName,strcat(dataDirCopy,buffer));

  meanVelSq=0;

  strcpy(dataDirCopyX,saveDirectory);
  snprintf(bufferX, 256,"velDistributionVx.dat");
  strcpy(fileNameX,strcat(dataDirCopyX,bufferX));

  strcpy(dataDirCopyY,saveDirectory);
  snprintf(bufferY, 256,"velDistributionVy.dat");
  strcpy(fileNameY,strcat(dataDirCopyY,bufferY));

   strcpy(dataDirCopyZ,saveDirectory);
  snprintf(bufferZ, 256,"velDistributionVz.dat");
  strcpy(fileNameZ,strcat(dataDirCopyZ,bufferZ));
  
  fa=fopen(fileName, "a");
  faX = fopen(fileNameX, "a");
  faY = fopen(fileNameY, "a");
  faZ = fopen(fileNameZ, "a");
  
  //strcpy(dataDirCopy2,saveDirectory);
  //snprintf(buffer2, 256,"velDistributionTime%lg.dat",t);
  //strcpy(fileName2,strcat(dataDirCopy2,buffer2));
  
  //fa=fopen(fileName, "a");
  //fa2=fopen(fileName2,"a");
  
  for (i=0;i<N0;i++){
    char dataDirCopy3[256];
    char buffer3[256];
    char fileName3[256];
    double x=X[0][i];
    double y=X[1][i];
    double z=X[2][i];
    //currVel=V[0][i];
    currWvFn = wvFns[i];
    popFg1DownmgMin1 += std::norm(currWvFn(0,0))/N0;
    popFg1Downmg0 += std::norm(currWvFn(1,0))/N0;
    popFg1Downmg1 += std::norm(currWvFn(2,0))/N0;
    popFg0mg0 += std::norm(currWvFn(3,0))/N0;
    popFg1UpmgMin1 += std::norm(currWvFn(4,0))/N0;
    popFg1Upmg0 += std::norm(currWvFn(5,0))/N0;
    popFg1Upmg1 += std::norm(currWvFn(6,0))/N0;
    popFg2mgMin2 += std::norm(currWvFn(7,0))/N0;
    popFg2mgMin1+= std::norm(currWvFn(8,0))/N0;
    popFg2mg0+= std::norm(currWvFn(9,0))/N0;
    popFg2mg1+= std::norm(currWvFn(10,0))/N0;
    popFg2mg2+= std::norm(currWvFn(11,0))/N0;
    
    popFe1me1 += std::norm(currWvFn(12,0))/N0;
    popFe1me0 += std::norm(currWvFn(13,0))/N0;
    popFe1meMin1 += std::norm(currWvFn(14,0))/N0;
    popFe0me0 += std::norm(currWvFn(15,0))/N0;
    densMatrix = currWvFn*currWvFn.t();
    meanVelSq += (V[0][i]*V[0][i]+V[1][i]*V[1][i]+V[2][i]*V[2][i])/(3*N0);
    vxDist = V[0][i];
    vyDist = V[1][i];
    vzDist = V[2][i];
    fprintf(faX,"%lg\t",vxDist);
    fprintf(faY,"%lg\t",vyDist);
    fprintf(faZ,"%lg\t",vzDist);
  }
  fprintf(faX,"\n");
  fprintf(faY,"\n");
  fprintf(faZ,"\n");
  fclose(faX);
  fclose(faY);
  fclose(faZ);

  char dataDirCopy4[256];
  char buffer4[256];
  char fileName4[256];
  FILE *fa4;
  strcpy(dataDirCopy4,saveDirectory);
  snprintf(buffer4, 256,"fluorCountsVTime.dat");
  strcpy(fileName4,strcat(dataDirCopy4,buffer4));
  fa4=fopen(fileName4,"a");
  fprintf(fa4,"%lg\t%d\n",t,currFluorCounts);
  fclose(fa4);
  
  fprintf(fa,"%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\n",t,popFg2mg2,popFg2mg1,popFg2mg0,popFg2mgMin1,popFg2mgMin2,popFg1Upmg1,popFg1Upmg0,popFg1UpmgMin1,popFg1Downmg1,popFg1Downmg0,popFg1DownmgMin1,popFg0mg0,popFe1me1,popFe1me0,popFe1meMin1,popFe0me0,meanVelSq);
  fclose(fa);
  //forceRecordX[counter]=forceX;
  //forceRecordY[counter]=forceY;
  //forceRecordZ[counter]=forceZ;
  //fclose(fa2);
  //forceRecordX[counter]=forceX;
  //forceRecordY[counter]=forceY;
  //forceRecordZ[counter]=forceZ;
  //FDotVRecord[counter]=FDotVNorm;
   
}

/***********************************************************************/
/*                                                                     */
/*  main routine                                                       */
/*                                                                     */
/***********************************************************************/

int main(int argc, char **argv)
{

  int nb_threads = omp_get_max_threads();
  printf(">> omp_get_max_thread()\n>> %i\n", nb_threads);
  //setup directory structure
  
  detuning=atof(argv[1]);
  detRaman = atof(argv[2]);
  satParam = atof(argv[3]);
  R21 = atof(argv[4]);
  firstState = (unsigned)atof(argv[5]);
  secondState = (unsigned)atof(argv[6]);
  CaOHOrCaFOrSrF = (unsigned)atof(argv[7]);
  double a;

  if (CaOHOrCaFOrSrF == 2){
    aParam = CaOHAParam;
    for(int i=0;i<4;i++){
      groundEnergies[i] = CaOHGroundEnergies[i];
    }
    vKick = vKickCaOH;
    gamOverC = CaOHGamOverC;
  }
     
  
  else if (CaOHOrCaFOrSrF == 1){
    aParam = CaFAParam;
    for(int i=0;i<4;i++){
      groundEnergies[i] = CaFGroundEnergies[i];
    }
    vKick = vKickCaF;
    gamOverC = CaFGamOverC;
  }
  
  else{
    aParam = SrFAParam; 
    for(int i=0;i<4;i++){
      groundEnergies[i] = SrFGroundEnergies[i];
    }
    vKick = vKickSrF;
    gamOverC = SrFGamOverC;
  }
  a=aParam;
  double b = sqrt(1-a*a);
		 
  Om = sqrt(satParam/2)*sqrt(1.0/(R21+1.0));
  OmRepump = sqrt(satParam/2)*sqrt(R21/(R21+1.0));
  detLaserDiff = groundEnergies[secondState]-groundEnergies[firstState]+detRaman;
  detShift = -1*groundEnergies[firstState];
  cout<<detLaserDiff;
  mkdir(saveDirectory,ACCESSPERMS);
  //make new sub directory
  char namebuf0[256];
  char namebuf[256];
  char namebuf2[256];
  char namebuf3[256];
  char namebuf4[256];
  char saveDirBackup[256];
  strcpy(saveDirBackup,saveDirectory);
  //sprintf(namebuf0,"Om%d/",(unsigned)(Om*100));
  //strcat(saveDirectory,namebuf0);
  //mkdir(saveDirectory,ACCESSPERMS);
  cout<<detRaman;
  sprintf(namebuf,"Det%dNumIons%dSatParam%lgDetRaman%lgR21%lgSecondState%dCaFOrSrF%d/",(unsigned)(detuning*100),(unsigned)N0,satParam,(detRaman*1000),R21,secondState,CaOHOrCaFOrSrF);
  strcat(saveDirectory,namebuf);
  mkdir(saveDirectory,ACCESSPERMS);
  cout<<"blah\n";
  //make directory for given IRat
  //sprintf(namebuf2,"IRat%lg/",1+2*floor((job-1.0)/13));
  //sprintf(namebuf2,"IRat%lg",IRat);
  //strcat(saveDirectory,namebuf2);
  //mkdir(saveDirectory,ACCESSPERMS);
    //make directory for given R21
  //sprintf(namebuf3,"RTwoOne%d/",(unsigned)(R21*100));
  //strcat(saveDirectory,namebuf3);
  //mkdir(saveDirectory,ACCESSPERMS);
      //make directory for given R21
  //sprintf(namebuf4,"detRaman%d/",(unsigned)(detRaman*100000));
  //strcat(saveDirectory,namebuf4);
  //mkdir(saveDirectory,ACCESSPERMS);
  //saveDirectory is now of form "OriginalSaveDirectory/Gamma%d...etc/job1/"
  
  //for SrF: First list sigPlus, then sigMinus, then Pi.
  //sigma+ coupling.  Makes a 'cs' matrix where the only non-zero term is a '1' with a row given by the ground state and column by the excited state
  cs[0] = wvFn1*wvFn14.t();//|F=1,J=1/2,m=-1> to |F'=1,m'=0> (J'=1/2 always so I don't list it.  For F=1 states, the J numbers here are the 'post-mixing' 'effective' J numbers)
  cs[1] = wvFn1*wvFn16.t();//|F=1,J=1/2,m=-1> to |F'=0,m'=0> 
  cs[2] = wvFn2*wvFn13.t();//|F=1,J=1/2,m=-0> to |F'=1,m'=+1> 
  cs[3] = wvFn4*wvFn13.t();//|F=0,J=1/2,m=0> to |F'=1,m'=+1> 
  cs[4] = wvFn5*wvFn14.t();//|F=1,J=3/2,m=-1> to |F'=1,m'=0> 
  cs[5] = wvFn5*wvFn16.t();//|F=1,J=3/2,m=-1> to |F'=0,m'=0> 
  cs[6] = wvFn6*wvFn13.t();//|F=1,J=3/2,m=0> to |F'=1,m'=+1> 
  cs[7] = wvFn8*wvFn15.t();//|F=2,J=3/2,m=-2> to |F'=1,m'=-1> 
  cs[8] = wvFn9*wvFn14.t();//|F=2,J=3/2,m=-1> to |F'=1,m'=0> 
  cs[9] = wvFn9*wvFn16.t();//|F=2,J=3/2,m=-1> to |F'=0,m'=0> 
  cs[10] = wvFn10*wvFn13.t();//|F=2,J=3/2,m=0> to |F'=1,m'=+1> 

  //sigma-
  cs[11] = wvFn2*wvFn15.t();//|F=1,J=1/2,m=0> to |F'=1,m'=-1> 
  cs[12] = wvFn3*wvFn14.t();//|F=1,J=1/2,m=+1> to |F'=1,m'=0> 
  cs[13] = wvFn3*wvFn16.t();//|F=1,J=1/2,m=+1> to |F'=0,m'=0> 
  cs[14] = wvFn4*wvFn15.t();//|F=0,J=1/2,m=0> to |F'=1,m'=-1> 
  cs[15] = wvFn6*wvFn15.t();//|F=1,J=3/2,m=0> to |F'=1,m'=-1> 
  cs[16] = wvFn7*wvFn14.t();//|F=1,J=3/2,m=+1> to |F'=1,m'=0> 
  cs[17] = wvFn7*wvFn16.t();//|F=1,J=3/2,m=+1> to |F'=0,m'=0> 
  cs[18] = wvFn10*wvFn15.t();//|F=2,J=3/2,m=0> to |F'=1,m'=-1> 
  cs[19] = wvFn11*wvFn14.t();//|F=2,J=3/2,m=+1> to |F'=1,m'=0> 
  cs[20] = wvFn11*wvFn16.t();//|F=2,J=3/2,m=+1> to |F'=0,m'=0> 
  cs[21] = wvFn12*wvFn13.t();//|F=2,J=3/2,m=+2> to |F'=1,m'=+1> 

  //pi
  cs[22] = wvFn1*wvFn15.t();//|F=1,J=1/2,m=-1> to |F'=1,m'=-1> 
  cs[23] = wvFn2*wvFn14.t();//|F=1,J=1/2,m=0> to |F'=1,m'=-0> 
  cs[24] = wvFn2*wvFn16.t();//|F=1,J=1/2,m=0> to |F'=0,m'=0> 
  cs[25] = wvFn3*wvFn13.t();//|F=1,J=1/2,m=+1> to |F'=1,m'=+1> 
  cs[26] = wvFn4*wvFn14.t();//|F=0,J=1/2,m=0> to |F'=1,m'=0> 
  cs[27] = wvFn4*wvFn16.t();//|F=0,J=1/2,m=0> to |F'=0,m'=0> 
  cs[28] = wvFn5*wvFn15.t();//|F=1,J=3/2,m=-1> to |F'=1,m'=-1> 
  cs[29] = wvFn6*wvFn14.t();//|F=1,J=3/2,m=0> to |F'=1,m'=0> 
  cs[30] = wvFn6*wvFn16.t();//|F=1,J=3/2,m=0> to |F'=0,m'=0> 
  cs[31] = wvFn7*wvFn13.t();//|F=1,J=3/2,m=+1> to |F'=1,m'=+1> 
  cs[32] = wvFn9*wvFn15.t();//|F=2,J=3/2,m=-1> to |F'=1,m'=-1> 
  cs[33] = wvFn10*wvFn14.t();//|F=2,J=3/2,m=0> to |F'=1,m'=0> 
  cs[34] = wvFn10*wvFn16.t();//|F=2,J=3/2,m=0> to |F'=0,m'=0> 
  cs[35] = wvFn11*wvFn13.t();//|F=2,J=3/2,m=+1> to |F'=1,m'=+1> 
  
  //corresponding rates, see Appendix to mainOBEWriteup.pdf
  //gs[0]=-0.4952;
  //gs[1]=-0.2653;
  //gs[2]=-0.4952;
  gs[0] = -sqrt(2.0)/3.0*a-b/6.0;
  gs[1] = -sqrt(2.0)/3.0*a+b/3.0;
  gs[2] = -sqrt(2.0)/3.0*a-b/6.0;
  gs[3]=sqrt(2.0)/3.0;
  //gs[4]=-0.0688;
  //gs[5]=-0.5128;
  //gs[6]=-0.0688;
  gs[4] = a/6.0-sqrt(2.0)/3.0*b;
  gs[5] = -a/3.0-sqrt(2.0)/3.0*b;
  gs[6] = a/6.0-sqrt(2.0)/3.0*b;
  gs[7]=-1./sqrt(6);
  gs[8]=-1./2./sqrt(3);
  gs[9]=0;
  gs[10]=-1./6;
  
  //gs[11]=0.4952;
  //gs[12]=0.4952;
  //gs[13]=-0.2653;
  gs[11] = -gs[0];
  gs[12] = -gs[0];
  gs[13] = gs[1];
  gs[14]=sqrt(2.)/3;
  //gs[15]=0.0688;
  //gs[16]=0.0688;
  //gs[17]=-0.5128;
  gs[15]= -gs[4];
  gs[16]= -gs[4];
  gs[17]= gs[5];
  gs[18]=-1./6;
  gs[19]=-1./2./sqrt(3);
  gs[20]=0;
  gs[21]=-1./sqrt(6);
  
  //gs[22]=0.4952;
  gs[23]=0;
  //gs[24]=-0.2653;
  //gs[25]=-0.4952;
  gs[22] = -gs[0];
  gs[24] = gs[1];
  gs[25] = gs[0];
  gs[26]=-sqrt(2.)/3;
  gs[27]=0;
  //gs[28]=0.0688;
  gs[29]=0;
  //gs[30]=-0.5128;
  //gs[31]=-0.0688;
  gs[28] = -gs[4];
  gs[30] = gs[5];
  gs[31] = gs[4];
  gs[32]=-1./2./sqrt(3);
  gs[33]=-1./3;
  gs[34]=0;
  gs[35]=-1./2./sqrt(3);

  /* 
  //Commented-out code here re-normalizes decay rate for cases where decay to certain states are ignored 
  if(onlyFg1DownToFe1){
    for(int i=0;i<36;i++){
      if(i==0||i==2||i==11||i==12||i==22||i==23||i==25){
	gs[i]=gs[i]*2.07;
      }
      else{
	gs[i]=0;
      }
      cout<<gs[i]<<"\n";

    }

  }

  else if(onlyFg1DownToFe0){
    for(int i=0;i<36;i++){
      if(i==1||i==13||i==24){
      	gs[i]=gs[i]*1.01;
      }
      else{
	gs[i]=0;
      }
    }
  }

  else if(onlyFg1UpToFe1){
    detShift = -19.6;
    for(int i=0;i<36;i++){
      if(i==4||i==6||i==15||i==16||i==28||i==29||i==31){
	gs[i]=gs[i]*1.94;
      }
      else{
	gs[i]=0;
      }

    }

  }

  else if(onlyFg1UpToFe0){
    detShift = -19.6;
    for(int i=0;i<36;i++){
      if(i==5||i==17||i==30){
      	gs[i]=gs[i]*7.28;
      }
      else{
	gs[i]=0;
      }
    }
  }

  else if(onlyFg0ToFe1){
    detShift = -7.5;
    for(int i=0;i<36;i++){
      if(i==3||i==14||i==26){
	gs[i]=gs[i]*2.12;
      }
      else{
	gs[i]=0;
      }

    }

  }

  else if(onlyFg2ToFe1){
    detShift = -25.9;
    for(int i=0;i<36;i++){
      if(i==8||i==10||i==7||i==18||i==19||i==21||i==32||i==35||i==33){
      	gs[i]=gs[i]*1.8974;
      }
      else{
	gs[i]=0;
      }
    }
  }

  //remove hyperfine levels as decided

  if(removeFg1Down){
    for(int i=0;i<36;i++){
      if(i!=0 && i!=2 && i!=11 && i!=12 && i!=22 && i!=23 && i!=25 && i!=1 && i!=13 && i!=24 && i!=5 && i!=17 && i!=30){
	//cout<<gs[i]<<"\n";
	gs[i] = gs[i]*1.0/sqrt(1.0-2.0*gs[0]*gs[0]);
	//cout<<gs[i]<<"\n";
      }
      
      else if(i==5||i==17||i==30){
	gs[i] = gs[i]/sqrt(1.0-3.0*gs[13]*gs[13]);
      }
    }
    
    for(int i=0;i<36;i++){
      if(i==0||i==2||i==11||i==12||i==22||i==23||i==25||i==1||i==13||i==24){
	gs[i]=gs[i]*0;
      }
    }
      
  }
      


  if(removeFg0){
    for(int i=0;i<36;i++){
      if(i!=3 && i!=14 && i!=26 && i!=5 && i!=17 && i!=30 && i!=1 && i!=13 && i!=24){
	gs[i] = gs[i]/sqrt(1.0-gs[3]*gs[3]);
      }
      else if(i==5||i==17||i==30||i==1||i==13||i==24){
	gs[i] = gs[i];
      }
      
    }
    for(int i=0;i<36;i++){

      if(i==3||i==14||i==26){
	gs[i]=gs[i]*0;
      }
    }
      
  }

  if(removeFg1Up){
    for(int i=0;i<36;i++){
      if(i!=4 && i!=6 && i!=15 && i!=16 && i!=28 && i!=29 && i!=31 && i!=5 && i!=17 && i!=30 && i!=1 && i!=13 && i!=24){
	gs[i] = gs[i]/sqrt(1.0-2.0*gs[4]*gs[4]);
      }
      
      else if(i==1||i==13||i==24){
	gs[i] = gs[i]/sqrt(1-3.0*gs[5]*gs[5]);
      }
      
    }
    for(int i=0;i<36;i++){
      if(i==4||i==6||i==15||i==16||i==28||i==29||i==31||i==5||i==17||i==30){
	gs[i]=gs[i]*0;
      }
    }
      
  }

  if(removeFg2){
    for(int i=0;i<36;i++){
      if(i!=8 && i!=10 && i!=7 && i!=18 && i!=19 && i!=21 && i!=32 && i!=35 && i!=33 && i!=5 && i!=17 && i!=30 && i!=1 && i!=13 && i!=24){
	gs[i] = gs[i]/sqrt(1.0-gs[10]*gs[10]-gs[21]*gs[21]-gs[35]*gs[35]);
      }
      
      else if(i==5||i==17||i==30||i==1||i==13||i==24){
	gs[i] = gs[i];
      }
      
    }

    for(int i=0;i<36;i++){
      if(i==8||i==10||i==7||i==18||i==19||i==21||i==32||i==35||i==33){
	gs[i]=gs[i]*0;
      }

    }
      
  }
  */
  cout<<"F1mf0 decay "<<gs[0]*gs[0]+gs[4]*gs[4]+gs[8]*gs[8]+gs[12]*gs[12]+gs[16]*gs[16]+gs[19]*gs[19]+gs[23]*gs[23]+gs[26]*gs[26]+gs[29]*gs[29]+gs[33]*gs[33]<<"\n";
  cout<<"F1mfMin1 decay "<<gs[2]*gs[2]+gs[3]*gs[3]+gs[6]*gs[6]+gs[10]*gs[10]+gs[21]*gs[21]+gs[25]*gs[25]+gs[31]*gs[31]+gs[35]*gs[35]<<"\n";
  cout<<"F1mf1 decay "<<gs[7]*gs[7]+gs[11]*gs[11]+gs[14]*gs[14]+gs[15]*gs[15]+gs[18]*gs[18]+gs[22]*gs[22]+gs[28]*gs[28]+gs[32]*gs[32]<<"\n";
    cout<<"F0mf0 decay "<<gs[1]*gs[1]+gs[5]*gs[5]+gs[9]*gs[9]+gs[13]*gs[13]+gs[17]*gs[17]+gs[20]*gs[20]+gs[24]*gs[24]+gs[27]*gs[27]+gs[30]*gs[30]+gs[34]*gs[34]<<"\n";

    //write hamiltonian terms: energy term accounts for detuning of laser ('given' to the excited state here) and the energy of each ground state (detShift=0 usually, defining 0 of energy to be |F=1,J=1/2> state)
    //Raman detuning comes into play
  hamEnergyTerm = -detuning*(wvFn13*wvFn13.t()+wvFn14*wvFn14.t()+wvFn15*wvFn15.t()+wvFn16*wvFn16.t());//energy terms are of form "Energy" X |n\rangle \langle
  hamEnergyTerm += (detShift+groundEnergies[0])*(wvFn1*wvFn1.t()+wvFn2*wvFn2.t()+wvFn3*wvFn3.t())+(detShift+groundEnergies[1])*wvFn4*wvFn4.t()+(detShift+groundEnergies[2])*(wvFn5*wvFn5.t()+wvFn6*wvFn6.t()+wvFn7*wvFn7.t())+(detShift+groundEnergies[3])*(wvFn8*wvFn8.t()+wvFn9*wvFn9.t()+wvFn10*wvFn10.t()+wvFn11*wvFn11.t()+wvFn12*wvFn12.t());   
	
  //now factor in the decay type terms, see "QTOBEWriteup.pdf"
  for(int j=0;j<36;j++){
    hamDecayTerm=hamDecayTerm-1./2*I*(gs[j]*gs[j]*cs[j].t()*cs[j]);
    decayMatrix = decayMatrix + gs[j]*gs[j]*cs[j].t()*cs[j];
  }

  //Atom-laser coupling terms
  for(int k=0;k<11;k++){
    hamCouplingTermOnlySigPlus+=cs[k].t()*gs[k];
  }
  for(int k=11;k<22;k++){
    hamCouplingTermOnlySigMinus+=cs[k].t()*gs[k];
  }
  for(int k=22;k<36;k++){
    hamCouplingTermOnlyPi+=cs[k].t()*gs[k];
  }

  
  srand48((unsigned)time(NULL)); // initialize random number generator

  int counter=0;
  t=0;
  dt=TIMESTEP;
  c0=0;
  init();

  while(t<=tmax)                     // run simulation until tmax
    {
      t+=dt;
      // other outputs like energy, vel_dist
      if ((c0+1)%sampleFreq == 0)
	{
	  //output();
	  outputData(counter);
	  //cout<<currFluorCounts<<"\n";
	  currFluorCounts = 0;
	  cout<<counter<<"\n";
	  //counter++;
	  //cout<<X[2][0]<<"\n";
	}

      
      qstep(c0);
      step();
      c0++;
    }




  return 0;
}
