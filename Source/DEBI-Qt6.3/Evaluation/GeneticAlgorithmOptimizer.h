/*!
* \file
* This file is part of the Evaluation module.
*
* \remarks
* 
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#pragma once

#include <Evaluation/Export.h>
#include <Evaluation/AbstractModel.h>
#include <Evaluation/AbstractAnalytics.h>
#include <Evaluation/DEBINN.h>
#include <Evaluation/DataPackage.h>
#include <QMap>
#include <QVariant>
#include <QBuffer>

namespace muw
{

//-----------------------------------------------------------------------------

/*!
* \brief The GeneticAlgorithmOptimizer class models evolutionary processes to train DEBI neural networks. 
*
* \details The training randomly initilizes a population of DEBI neural network instances and measures their fitness.
* The training iterations further evolve the previous iterations by creating new offsprings. An offspring is created by crossover and mutation of parents selected through tournament selection.
* The final result is a parameter set of a trained DEBI neural network either selected as the fittest member in the last iteration or as an ensemble of the fittest k (k>1) members.
* The training process is logged which can be read out after the trianing is complete. In case an optional validation set is provided, its performance is measured and stored in the log container next to 
* training performances across iterations.
*/

class Evaluation_API GeneticAlgorithmOptimizer
{

public:

	/*!
	\brief Constructor.
	\param [in] aDEBINN the DEBI neural network to train based on aDataPackage.
	\param [in] aDataPackage the trianing data package used to train the input aDEBINN.
	\param [in] aSettings the settings container for the training process.
	\param [in] aValidationDataPackage the optional input to log the performance of aDEBINN during training next to its training performance. This data pacakge does not take part in any decision making processes during the training.
	*/
	GeneticAlgorithmOptimizer( muw::DEBINN* aDEBINN, muw::DataPackage* aDataPackage, QVariantMap aSettings, muw::DataPackage* aValidationDataPackage );

	/*!
	\brief Destructor.
	*/
	virtual ~GeneticAlgorithmOptimizer();

	/*!
	\brief Trains the DEBI neural network provided in the constructor.
	*/
	void build();

	/*!
	\return the list of trained parameters of the trained DEBI neural network.
	*/
	QVector< double > result();

	/*!
	\return the log container created during the trianing process.
	*/
	QVariantMap log() { return mLog; }

private:

	/*!
	\brief Randomly splits the training datapackage and assigns an analytics entity to each of them. This is done for parallelization purposes.
	*/
	void generateAnalytics( int aCurrentIteration );

	/*!
	\brief Randomly shuffles the analytics entities, thus, models an internal training subset shuffle across the training iterations.
	*/
	void shuffleDataPackages();

	/*!
	\brief Initializes the optimizer by creating a population of randomly created members.
	*/
	void initialize();

	/*!
	\brief Iterates the training by modeling evolutionary principles.
	*/
	void evolve();

	/*!
	\brief Randomly splits the current population to two subgroups and selects the fittest from each.
	\return the fittest members of the tournament selection which will serve as parents for an offspring member to be created.
	*/
	QPair< std::shared_ptr< muw::DEBINN >, std::shared_ptr< muw::DEBINN > > tournamentSelection();

	/*!
	\brief Creates an offspring from the parents by modeling crossover and mutation.
	\param [in] aParent1 the first parent to create the offspring.
	\param [in] aParent2 the second parent to create the offspring.
	\return the offspring member of the crossover and mutation.
	*/
	std::shared_ptr< muw::DEBINN > offspring( std::shared_ptr< muw::DEBINN > aParent1, std::shared_ptr< muw::DEBINN >  aParent2, unsigned int aRandomSeed );

	/*!
	\brief Creates a random offspring. Used in the initializaion stage of the trianing.
	\return the offspring.
	*/
	std::shared_ptr< muw::DEBINN > offspring( unsigned int aRandomSeed );

	/*!
	\brief Tests if early stopping shall be performed.
	\return true if early stopping is activated in the given iteration. Returns false otherwise.
	*/
	bool isEarlyStopping( int aCurrentIteration );


private:

	muw::DEBINN*                                         mAdam;                                //!< The initially-provided DEBI neural network to read out its basic properties for creating all other members during training.
	muw::DataPackage*                                    mDataPackage;                         //!< The data package to train the DEBI neural network.
	muw::DataPackage*                                    mValidationDataPackage;               //!< The data package to validate the DEBI neural network.
	bool                                                 mIsExternalValidation;                //!< Stores if an external validation set is provided. If not provided, the training dataset will be split to internal train-validate subsets.
	muw::AbstractAnalytics*                              mValidationAnalytics;                 //!< The dalidation data package. Optional, purely for logging purposes.
	QVariantMap                                          mLog;                                 //!< The log container, having the training and validation (optional) performance values of training iterations.
	QVector< muw::AbstractAnalytics* >                   mAnalyticsTrain;                      //!< The analytics classes that provide the performance values for each input DEBI model candidate as of their internal datasets.
	QString                                              mAnalyticsType;                       //!< The type of analytics to rely on for training.
	QString                                              mAnalyticsUnit;                       //!< The unit of the analytics.
	double                                               mAlpha;                               //!< The alpha value to weigh correct and incorrect (1-mAlpha) predictions in case an entropy loss analytics is utilized.
	QMultiMap< double, std::shared_ptr< muw::DEBINN > >  mPopulation;                          //!< The population, containing DEBI neural network instances to model evolutionary processes.
	QMultiMap< double, std::shared_ptr< muw::DEBINN > >  mHallOfFame;                          //!< The population, containing the historically best-performing members across training as of the validation dataset. The final model wil be selected from here.
	int                                                  mLastAlphaIteration;                  //!< Stores in which iteration the alpha model was created.
	double                                               mMaximumMutationRate;                 //!< The maximum mutation rate at the start of the trianing.
	double                                               mMinimumMutationRate;                 //!< The minimum mutation rate at the end of the training.
	double                                               mCurrentMutationRate;                 //!< The mutation rate in the given iteration.
	double                                               mMutationDelta;                       //!< The chance a digital gene is mutated.
	int                                                  mPopulationCount;                     //!< The number of DEBI neural network instances in the population.
	int                                                  mIterationCount;                      //!< The number of iterations to evolve the population.
	int                                                  mEnsembleCount;                       //!< The number of final DEBI instances to average as output of the training.
	std::mt19937*                                        mGenerator;                           //!< Random generator for modeling random events during the training.
	QVector< double >                                    mResult;                              //!< The trained parameters of the final DEBI neural network.
	bool                                                 mIsInternalSubsets;                   //!< Stores if the training shall utilize internal randomly-splitted training subsets or use the input dataset as is.
	QString                                              mGenerationTraining;                  //!< The method to handle training in-between populations over iterations.
	int                                                  mRegenerateAnalyticsFrequency;        //!< Stores how many iterations the last random train-validate splits shall be stored. Afterwards, a new random split is created.
	int                                                  mEarlyStoppingMargin;                 //!< Stores how many iterations the algorithm shall still attempt to find a better model, in case an early stipping model candidate was identified.
	double                                               mTrainValidateTolerance;              //!< Stores the fitness tolerance between train and validate fitnesses. Only models having a train-validate fitness difference less than the tolerance will be considered ideal to return with.
	int                                                  mFinalModelSelectionAnalyticsCount;   //!< Stores how many analytics (random subsamples) need to be created to test the Hall of Fame members.
	QString                                              mFinalModelSelectionMethod;           //!< The method to select the final model from the Hall of Fame container.
	bool                                                 mIsAlphaToSavePerIteration;           //!< Stores if the highest-performing ampha model variant in each iteration shall be saved to a model folder.
	QString                                              mModelSaveTargetFolder;               //!< The absolute path to the folder save model variants into during training.
};

//-----------------------------------------------------------------------------

}
