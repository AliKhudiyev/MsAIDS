
/**
 This is program entry for STD template for sinusit

*/


#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <CLogger.h>
#include "COptionParser.h"
#include "CRandomGenerator.h"
#include "CEvolutionaryAlgorithm.h"
#include "global.h"
#include "sinusitIndividual.hpp"

using namespace std;

/** Global variables for the whole algorithm */
CIndividual** pPopulation = NULL;
CIndividual*  bBest = NULL;
float* pEZ_MUT_PROB = NULL;
float* pEZ_XOVER_PROB = NULL;
unsigned *EZ_NB_GEN;
unsigned *EZ_current_generation;
int EZ_POP_SIZE;
int OFFSPRING_SIZE;

std::vector<char *> vArgv;

CEvolutionaryAlgorithm* EA;

int main(int argc, char** argv){

	/* Let's give an access to varible argv for users and put it to vArgv */
	if (argc > 1){
    	    for (int i = 1; i < argc; i++){
        	if ((argv[i][0]=='-')&&(argv[i][1]=='-')) break;
            	    vArgv.push_back(argv[i]);
    	    }
        }
       

	parseArguments("sinusit.prm",argc,argv);

	ParametersImpl p;
	p.setDefaultParameters(argc,argv);
	CEvolutionaryAlgorithm* ea = p.newEvolutionaryAlgorithm();

	EA = ea;

	sinusitInit(argc,argv);

	CPopulation* pop = ea->getPopulation();

	ea->runEvolutionaryLoop();

	sinusitFinal(pop);

	delete pop;


	return 0;
}

