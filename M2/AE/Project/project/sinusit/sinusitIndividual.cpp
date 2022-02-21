

#include <fstream>
#include <time.h>
#include <cstring>
#include <sstream>
#include "CRandomGenerator.h"
#include "CPopulation.h"
#include "COptionParser.h"
#include "CStoppingCriterion.h"
#include "CEvolutionaryAlgorithm.h"
#include "global.h"
#include "CIndividual.h"

using namespace std;
bool bReevaluate = false;
#include "sinusitIndividual.hpp"
bool INSTEAD_EVAL_STEP = false;

CRandomGenerator* globalRandomGenerator;
extern CEvolutionaryAlgorithm* EA;

#define STD_TPL

// User declarations
#line 6 "sinusit.ez"

#define NB_SIN 3  //# of sines
#define AMP 100.0 // max val for amplitude
#define FREQ 10.0 // max val for frequency
#define PH 6.28   // max val for phase
#define X 100.0 // max for x values
#define NB_SAMP 1000 //# of samples


float fSamp[NB_SAMP][2]; //for x and y values
float fSin[NB_SIN*3]; //each sine has 3 parameters





// User functions

#line 21 "sinusit.ez"



// Initialisation function
void EASEAInitFunction(int argc, char *argv[]){
#line 26 "sinusit.ez"


//if we want to globalRandomGenerator->randomly create the sines
for (int s=0;s<NB_SIN;s++){
	fSin[s*3+0]=globalRandomGenerator->random(0.0,AMP);
	fSin[s*3+1]=globalRandomGenerator->random(0.0,FREQ);
	fSin[s*3+2]=globalRandomGenerator->random(0.0,PH);
}

/*
If we want to define specific sines
fSin[0*3+0]=20; fSin[0*3+1]=7; fSin[0*3+2]=2; the first sine
fSin[1*3+0]=70; fSin[1*3+1]=5; fSin[1*3+2]=1; the second sine
*/

for (int n=0;n<NB_SAMP;n++){
	float fSum=0.0, x=fSamp[n][0]=globalRandomGenerator->random(0.0,X);
	for (int s=0;s<NB_SIN;s++) fSum+=fSin[s*3+0]*sin(fSin[s*3+1]*x+fSin[s*3+2]);
	fSamp[n][1]=fSum;
}

}

// Finalization function
void EASEAFinalization(CPopulation* population){
#line 49 "sinusit.ez"


printf("Functions looked for: y=");
for (int s=0; s<NB_SIN; s++) printf("%f*sin(%f*x+%f)+",fSin[s*3+0],fSin[s*3+1],fSin[s*3+2]);
printf("\n Function obtained : y=");
for (int s=0; s<NB_SIN; s++) printf("%f*sin(%f*x+%f)+",((IndividualImpl*)bBest)->fSin[s*3+0], ((IndividualImpl*)bBest)->fSin[s*3+1], ((IndividualImpl*)bBest)->fSin[s*3+2]);
printf("\n\n");

}



void evale_pop_chunk(CIndividual** population, int popSize){
  
// No Instead evaluation step function.

}

void sinusitInit(int argc, char** argv){
	
  EASEAInitFunction(argc, argv);

}

void sinusitFinal(CPopulation* pop){
	
  EASEAFinalization(pop);
;
}

void EASEABeginningGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
	#line 155 "sinusit.ez"
{
#line 59 "sinusit.ez"

}
}

void EASEAEndGenerationFunction(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
	{

}
}

void EASEAGenerationFunctionBeforeReplacement(CEvolutionaryAlgorithm* evolutionaryAlgorithm){
	{

}
}


IndividualImpl::IndividualImpl() : CIndividual() {
   
  // Genome Initialiser
#line 77 "sinusit.ez"
 // "initializer" is also accepted
for(int s=0; s<NB_SIN; s++ ){
   (*this).fSin[s*3+0]=globalRandomGenerator->random(0.0,AMP);
   (*this).fSin[s*3+1]=globalRandomGenerator->random(0.0,FREQ);
   (*this).fSin[s*3+2]=globalRandomGenerator->random(0.0,PH);
}

  valid = false;
  isImmigrant = false;
}

CIndividual* IndividualImpl::clone(){
	return new IndividualImpl(*this);
}

IndividualImpl::~IndividualImpl(){
  // Destructing pointers

}


float IndividualImpl::evaluate(){
  if(valid)
    return fitness;
  else{
    valid = true;
    #line 107 "sinusit.ez"
 // Returns the score
  float fScore=0;
  
  for (int n=0;n<NB_SAMP;n++){
  	float x=fSamp[n][0], y=fSamp[n][1], fSum=0.0;
  	for (int s=0;s<NB_SIN;s++)
  	fSum+=(*this).fSin[s*3+0]*sin((*this).fSin[s*3+1]*x+(*this).fSin[s*3+2]);
  	fScore+=fabs(fSum-y);
  }
   
  return fitness =  fScore;

  }
}

void IndividualImpl::boundChecking(){
	
// No Bound checking function.

}

string IndividualImpl::serialize(){
    ostringstream EASEA_Line(ios_base::app);
    // Memberwise serialization
	for(int EASEA_Ndx=0; EASEA_Ndx<9; EASEA_Ndx++)
		EASEA_Line << this->fSin[EASEA_Ndx] <<" ";

    EASEA_Line  << this->fitness;
    return EASEA_Line.str();
}

void IndividualImpl::deserialize(string Line){
    istringstream EASEA_Line(Line);
    string line;
    // Memberwise deserialization
	for(int EASEA_Ndx=0; EASEA_Ndx<9; EASEA_Ndx++)
		EASEA_Line >> this->fSin[EASEA_Ndx];

    EASEA_Line >> this->fitness;
    this->valid=true;
    this->isImmigrant = false;
}

IndividualImpl::IndividualImpl(const IndividualImpl& genome){

  // ********************
  // Problem specific part
  // Memberwise copy
    {for(int EASEA_Ndx=0; EASEA_Ndx<9; EASEA_Ndx++)
       fSin[EASEA_Ndx]=genome.fSin[EASEA_Ndx];}



  // ********************
  // Generic part
  this->valid = genome.valid;
  this->fitness = genome.fitness;
  this->isImmigrant = false;
}


CIndividual* IndividualImpl::crossover(CIndividual** ps){
	// ********************
	// Generic part
	IndividualImpl** tmp = (IndividualImpl**)ps;
	IndividualImpl parent1(*this);
	IndividualImpl parent2(*tmp[0]);
	IndividualImpl child(*this);

	//DEBUG_PRT("Xover");
	/*   cout << "p1 : " << parent1 << endl; */
	/*   cout << "p2 : " << parent2 << endl; */

	// ********************
	// Problem specific part
  	#line 85 "sinusit.ez"


  int nLocus=globalRandomGenerator->random(1,NB_SIN-1);
  for (int i=nLocus;i<NB_SIN;i++){
  	child.fSin[i*3+0]=parent2.fSin[i*3+0];
  	child.fSin[i*3+1]=parent2.fSin[i*3+1];
  	child.fSin[i*3+2]=parent2.fSin[i*3+2];
  }




	child.valid = false;
	/*   cout << "child : " << child << endl; */
	return new IndividualImpl(child);
}


void IndividualImpl::printOn(std::ostream& os) const{
	

}

std::ostream& operator << (std::ostream& O, const IndividualImpl& B)
{
  // ********************
  // Problem specific part
  O << "\nIndividualImpl : "<< std::endl;
  O << "\t\t\t";
  B.printOn(O);

  if( B.valid ) O << "\t\t\tfitness : " << B.fitness;
  else O << "fitness is not yet computed" << std::endl;
  return O;
}


void IndividualImpl::mutate( float pMutationPerGene ){
  this->valid=false;


  // ********************
  // Problem specific part
  #line 96 "sinusit.ez"


float fProbMutPerGene=1.0/(float)(NB_SIN*3);
for (int s=0; s<NB_SIN;s++){
	if(globalRandomGenerator->tossCoin(fProbMutPerGene)) (*this).fSin[s*3+0]+=globalRandomGenerator->random(-AMP,AMP)/10.0;
	if(globalRandomGenerator->tossCoin(fProbMutPerGene)) (*this).fSin[s*3+1]+=globalRandomGenerator->random(-FREQ,FREQ)/10.0;
	if(globalRandomGenerator->tossCoin(fProbMutPerGene)) (*this).fSin[s*3+2]+=globalRandomGenerator->random(-PH,PH)/10.0;
}


}

void ParametersImpl::setDefaultParameters(int argc, char** argv){

	this->minimizing = false;
	this->nbGen = setVariable("nbGen",(int)100);
	this->nbCPUThreads = setVariable("nbCPUThreads", 1);
	this->reevaluateImmigrants = setVariable("reevaluateImmigrants", 0);

	omp_set_num_threads(this->nbCPUThreads);
	seed = setVariable("seed",(int)time(0));
	globalRandomGenerator = new CRandomGenerator(seed);
	this->randomGenerator = globalRandomGenerator;


	selectionOperator = getSelectionOperator(setVariable("selectionOperator","Tournament"), this->minimizing, globalRandomGenerator);
	replacementOperator = getSelectionOperator(setVariable("reduceFinalOperator","Tournament"),this->minimizing, globalRandomGenerator);
	parentReductionOperator = getSelectionOperator(setVariable("reduceParentsOperator","Tournament"),this->minimizing, globalRandomGenerator);
	offspringReductionOperator = getSelectionOperator(setVariable("reduceOffspringOperator","Tournament"),this->minimizing, globalRandomGenerator);
	selectionPressure = setVariable("selectionPressure",(float)2.000000);
	replacementPressure = setVariable("reduceFinalPressure",(float)2.000000);
	parentReductionPressure = setVariable("reduceParentsPressure",(float)2.000000);
	offspringReductionPressure = setVariable("reduceOffspringPressure",(float)2.000000);
	pCrossover = 1.000000;
	pMutation = 1.000000;
	pMutationPerGene = 0.05;

	parentPopulationSize = setVariable("popSize",(int)10000);
	offspringPopulationSize = setVariable("nbOffspring",(int)10000);


	parentReductionSize = setReductionSizes(parentPopulationSize, setVariable("survivingParents",(float)1.000000));
	offspringReductionSize = setReductionSizes(offspringPopulationSize, setVariable("survivingOffspring",(float)1.000000));

	this->elitSize = setVariable("elite",(int)1);
	this->strongElitism = setVariable("eliteType",(int)1);

	if((this->parentReductionSize + this->offspringReductionSize) < this->parentPopulationSize){
		printf("*WARNING* parentReductionSize + offspringReductionSize < parentPopulationSize\n");
		printf("*WARNING* change Sizes in .prm or .ez\n");
		printf("EXITING\n");
		exit(1);	
	} 
	if((this->parentPopulationSize-this->parentReductionSize)>this->parentPopulationSize-this->elitSize){
		printf("*WARNING* parentPopulationSize - parentReductionSize > parentPopulationSize - elitSize\n");
		printf("*WARNING* change Sizes in .prm or .ez\n");
		printf("EXITING\n");
		exit(1);	
	} 
	if(!this->strongElitism && ((this->offspringPopulationSize - this->offspringReductionSize)>this->offspringPopulationSize-this->elitSize)){
		printf("*WARNING* offspringPopulationSize - offspringReductionSize > offspringPopulationSize - elitSize\n");
		printf("*WARNING* change Sizes in .prm or .ez\n");
		printf("EXITING\n");
		exit(1);	
	} 
	

	/*
	 * The reduction is set to true if reductionSize (parent or offspring) is set to a size less than the
	 * populationSize. The reduction size is set to populationSize by default
	 */
	if(offspringReductionSize<offspringPopulationSize) offspringReduction = true;
	else offspringReduction = false;

	if(parentReductionSize<parentPopulationSize) parentReduction = true;
	else parentReduction = false;

	generationalCriterion = new CGenerationalCriterion(setVariable("nbGen",(int)100));
	controlCStopingCriterion = new CControlCStopingCriterion();
	timeCriterion = new CTimeCriterion(setVariable("timeLimit",0));

	this->optimise = 0;

	this->printStats = setVariable("printStats",1);
	this->generateCSVFile = setVariable("generateCSVFile",0);
	this->generatePlotScript = setVariable("generatePlotScript",0);
	this->generateRScript = setVariable("generateRScript",0);
	this->plotStats = setVariable("plotStats",1);
	this->printInitialPopulation = setVariable("printInitialPopulation",0);
	this->printFinalPopulation = setVariable("printFinalPopulation",0);
	this->savePopulation = setVariable("savePopulation",0);
	this->startFromFile = setVariable("startFromFile",0);

	this->outputFilename = (char*)"sinusit";
	this->plotOutputFilename = (char*)"sinusit.png";

	this->remoteIslandModel = setVariable("remoteIslandModel",0);
	std::string* ipFilename=new std::string();
	*ipFilename=setVariable("ipFile","ip.txt");

	this->ipFile =(char*)ipFilename->c_str();
	this->migrationProbability = setVariable("migrationProbability",(float)0.300000);
    this->serverPort = setVariable("serverPort",2929);
}

CEvolutionaryAlgorithm* ParametersImpl::newEvolutionaryAlgorithm(){

	pEZ_MUT_PROB = &pMutationPerGene;
	pEZ_XOVER_PROB = &pCrossover;
	//EZ_NB_GEN = (unsigned*)setVariable("nbGen",100);
	EZ_current_generation=0;
  EZ_POP_SIZE = parentPopulationSize;
  OFFSPRING_SIZE = offspringPopulationSize;

	CEvolutionaryAlgorithm* ea = new EvolutionaryAlgorithmImpl(this);
	generationalCriterion->setCounterEa(ea->getCurrentGenerationPtr());
	ea->addStoppingCriterion(generationalCriterion);
	ea->addStoppingCriterion(controlCStopingCriterion);
	ea->addStoppingCriterion(timeCriterion);	

	EZ_NB_GEN=((CGenerationalCriterion*)ea->stoppingCriteria[0])->getGenerationalLimit();
	EZ_current_generation=&(ea->currentGeneration);

	 return ea;
}

void EvolutionaryAlgorithmImpl::initializeParentPopulation(){
	if(this->params->startFromFile){
	  ifstream EASEA_File("sinusit.pop");
	  string EASEA_Line;
  	  for( unsigned int i=0 ; i< this->params->parentPopulationSize ; i++){
	  	  getline(EASEA_File, EASEA_Line);
		  this->population->addIndividualParentPopulation(new IndividualImpl(),i);
		  ((IndividualImpl*)this->population->parents[i])->deserialize(EASEA_Line);
	  }
	  
	}
	else{
  	  for( unsigned int i=0 ; i< this->params->parentPopulationSize ; i++){
		  this->population->addIndividualParentPopulation(new IndividualImpl(),i);
	  }
	}
        this->population->actualParentPopulationSize = this->params->parentPopulationSize;
}


EvolutionaryAlgorithmImpl::EvolutionaryAlgorithmImpl(Parameters* params) : CEvolutionaryAlgorithm(params){
	;
}

EvolutionaryAlgorithmImpl::~EvolutionaryAlgorithmImpl(){

}

