/*!
* \file
* Member class definitions of GeneticAlgorithmOptimizer. This file is part of the Evaluation module.
*
* \remarks
* 
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#include <Evaluation/GeneticAlgorithmOptimizer.h>
#include <Evaluation/LossAnalytics.h>
#include <Evaluation/ConfusionMatrixAnalytics.h>
#include <Evaluation/MultiSubsetFoldGenerator.h>
#include <DataRepresentation/Types.h>
#include <QDebug>
#include <omp.h>

extern bool isRandomSeed;

namespace muw
{

//-----------------------------------------------------------------------------

GeneticAlgorithmOptimizer::GeneticAlgorithmOptimizer( muw::DEBINN* aDEBINN, muw::DataPackage* aDataPackage, QVariantMap aSettings, muw::DataPackage* aValidationDataPackage )
:
	mAdam( aDEBINN ),
	mDataPackage( aDataPackage ),
	mValidationDataPackage( aValidationDataPackage ),
	mValidationAnalytics( nullptr ),
	mIsExternalValidation( false ),
	mLog(),
	mAnalyticsTrain(),
	mAnalyticsType(),
	mAnalyticsUnit(),
	mAlpha( 0.0 ),
	mPopulation(),
	mMaximumMutationRate( aSettings.value( "Optimizer/MaximumMutationRate" ).toDouble() ),
	mMinimumMutationRate( aSettings.value( "Optimizer/MinimumMutationRate" ).toDouble() ),
	mCurrentMutationRate( 0.0 ),
	mMutationDelta( aSettings.value( "Optimizer/MutationDelta" ).toDouble() ),
	mPopulationCount( aSettings.value( "Optimizer/PopulationCount" ).toInt() ),
	mIterationCount( aSettings.value( "Optimizer/IterationCount" ).toInt() ),
	mEnsembleCount( aSettings.value( "Optimizer/EnsembleCount" ).toInt() ),
	mGenerator( nullptr ),
	mResult(),
	mIsInternalSubsets( false ),
	mGenerationTraining(),
	mRegenerateAnalyticsFrequency( aSettings.value( "Optimizer/RegenerateAnalyticsFrequency" ).toInt() ),
	mEarlyStoppingMargin( aSettings.value( "Optimizer/EarlyStoppingMargin" ).toInt() ),
	mTrainValidateTolerance( aSettings.value( "Optimizer/TrainValidateTolerance" ).toDouble() ),
	mFinalModelSelectionAnalyticsCount( aSettings.value( "Optimizer/ModelSelectionAnalyticsCount" ).toInt() ),
	mFinalModelSelectionMethod( aSettings.value( "Optimizer/ModelSelectionMethod" ).toString() ),
	mIsAlphaToSavePerIteration( false ),
	mModelSaveTargetFolder( "" )  // IMPORTANT: If mIsAlphaToSavePerIteration = true, you must provide a valid absolute path here.
{
	if ( isRandomSeed )
	{
		std::random_device rd{};
		mGenerator = new std::mt19937{ rd() };
	}
	else
	{
		mGenerator = new std::mt19937{ 123 };
	}

	if ( aValidationDataPackage != nullptr ) mIsExternalValidation = true;

	QString isInternalSubsets = aSettings.value( "Optimizer/InternalSubsets" ).toString();
	if ( isInternalSubsets == "false" || isInternalSubsets == "" )
	{
		mIsInternalSubsets = false;
	}
	else
	{
		mIsInternalSubsets = true;
	}

	mAnalyticsType = aSettings.value( "Analytics/Type" ).toString();
	
	if ( mAnalyticsType == "ConfusionMatrixAnalytics" )
	{
		mAnalyticsUnit = aSettings.value( "Analytics/ConfusionMatrix/Unit" ).toString();
	}
	else if ( mAnalyticsType == "LossAnalytics" )
	{
		mAnalyticsUnit = aSettings.value( "Analytics/Loss/Unit" ).toString();
		mAlpha = aSettings.value( "Analytics/Loss/Alpha" ).toDouble();
	}

	mGenerationTraining = aSettings.value( "Optimizer/GenerationTraining" ).toString();
	
	generateAnalytics( 0 );
}

//-----------------------------------------------------------------------------

GeneticAlgorithmOptimizer::~GeneticAlgorithmOptimizer()
{
	mPopulation.clear();
	if ( mGenerator != nullptr )
	{
		delete mGenerator;
		mGenerator = nullptr;
	}

	for ( int i = 0; i < mAnalyticsTrain.size(); ++i )
	{
		delete mAnalyticsTrain[ i ]->dataPackage();
		delete mAnalyticsTrain[ i ];
		mAnalyticsTrain[ i ] = nullptr;
	}

	mAnalyticsTrain.clear();

	delete mValidationAnalytics;
	mValidationAnalytics = nullptr;

	mLog.clear();
}

//-----------------------------------------------------------------------------

void GeneticAlgorithmOptimizer::build()
{
	initialize();

	qDebug() << "Execution with" << mIterationCount << "iterations" << mAnalyticsType << ":" << mAnalyticsUnit << Qt::endl;

	QVariantList mutationRatesLog;
	QVariantList iterationsLog;
	for ( int i = 0; i < mIterationCount; ++i )
	{
		double left  = 1.0 - ( double( i ) / mIterationCount );
		double right = 1.0 - left;
		mCurrentMutationRate = ( left * mMaximumMutationRate ) + ( right * mMinimumMutationRate );
		mutationRatesLog.push_back( mCurrentMutationRate );
		iterationsLog.push_back( i + 1 );
	}
	mLog.insert( "Iteration", iterationsLog );
	mLog.insert( "MutationRate", mutationRatesLog );

	QVariantList trainingErrorLog;
	QVariantList validateErrorLog;

	for ( int i = 0; i < mIterationCount; ++i )
	{
		double left  = 1.0 - ( double( i ) / mIterationCount );
		double right = 1.0 - left;
		mCurrentMutationRate = ( left * mMaximumMutationRate ) + ( right * mMinimumMutationRate );
	
		if ( mValidationAnalytics == nullptr )
		{
			trainingErrorLog.push_back( mPopulation.firstKey() );
			if ( i % 1 == 0 ) qDebug() <<  i << ";" << mCurrentMutationRate << ";" << mPopulation.firstKey() << Qt::endl;
		}
		else
		{
			trainingErrorLog.push_back( mPopulation.firstKey() );
			auto validationError = mValidationAnalytics->evaluate( mPopulation.first().get() );
			validateErrorLog.push_back( validationError );
			if ( i % 1 == 0 )  qDebug () << i << ";" << mCurrentMutationRate << ";" << mPopulation.firstKey () << ";" << validationError << Qt::endl;
		}

		auto latestAlpha = mPopulation.first().get();
		if ( mIsAlphaToSavePerIteration ) latestAlpha->save( mModelSaveTargetFolder + "/Model-" + QString::number( i ).rightJustified( 5, '0' ) + ".bin" );  // Use this line if you want to save the most fit models of each population across iterations.

		if ( !isEarlyStopping( i ) )
		{
			evolve();
		}
		else
		{
			qDebug() << "Early stopping activated";

			for ( int l = i; l < mIterationCount; ++l )
			{
				trainingErrorLog.push_back( -1.0 );
				validateErrorLog.push_back( -1.0 );
			}

			break;
		}

		if ( i % mRegenerateAnalyticsFrequency == 0 ) generateAnalytics( i );
	}

	mLog.insert( "TrainingError", trainingErrorLog );
	if ( !validateErrorLog.isEmpty() ) mLog.insert( "ValidateError", validateErrorLog );
}

//-----------------------------------------------------------------------------

void GeneticAlgorithmOptimizer::generateAnalytics( int aCurrentIteration )
{
	for ( int i = 0; i < mAnalyticsTrain.size(); ++i )
	{
		delete mAnalyticsTrain[ i ]->dataPackage();
		delete mAnalyticsTrain[ i ];
		mAnalyticsTrain[ i ] = nullptr;
	}

	mAnalyticsTrain.clear();

	if ( mValidationAnalytics != nullptr )
	{
		delete mValidationAnalytics->dataPackage();
		delete mValidationAnalytics;
		mValidationAnalytics = nullptr;
	}

	muw::DataPackage* trainingDataPackage  = nullptr;

	if ( !mIsExternalValidation )
	{
		if ( mValidationDataPackage != nullptr )
		{
			delete mValidationDataPackage;
			mValidationDataPackage = nullptr;
		}

		QMap< QString, double > subsetDefinitions;

		subsetDefinitions[ "Balanced" ] = 0.3;
		subsetDefinitions[ "Stratified" ] = -1.0;

		muw::MultiSubsetFoldGenerator MSFG( mDataPackage, subsetDefinitions, mIterationCount / mRegenerateAnalyticsFrequency, aCurrentIteration );
		MSFG.execute();

		auto subsets = MSFG.fold( int( aCurrentIteration / mRegenerateAnalyticsFrequency ) );
		trainingDataPackage    = subsets.value( "Stratified" );
		mValidationDataPackage = subsets.value( "Balanced" );
	}
	else
	{
		trainingDataPackage = mDataPackage;
	}

	if ( mValidationDataPackage != nullptr )
	{
		if ( mAnalyticsType == "ConfusionMatrixAnalytics" )
		{
			mValidationAnalytics = new muw::ConfusionMatrixAnalytics( new muw::DataPackage( mValidationDataPackage->featureDB(), mValidationDataPackage->labelDB(), mValidationDataPackage->labelName() ), mAnalyticsUnit );
		}
		else if ( mAnalyticsType == "LossAnalytics" )
		{
			mValidationAnalytics = new muw::LossAnalytics( new muw::DataPackage( mValidationDataPackage->featureDB(), mValidationDataPackage->labelDB(), mValidationDataPackage->labelName() ), mAnalyticsUnit, mAlpha );
		}
	}
	
	if ( mIsInternalSubsets )
	{
		QMap< QString, double > subsetDefinitions;

		subsetDefinitions[ "Balanced" ] = 0.2;
		subsetDefinitions[ "Stratified" ] = -1.0;

		muw::MultiSubsetFoldGenerator internalMSFG( trainingDataPackage, subsetDefinitions, mPopulationCount, aCurrentIteration );
		internalMSFG.execute();

		for ( int i = 0; i < mPopulationCount; ++i )
		{
			auto dataPackages = internalMSFG.fold( i );
			auto trainDP = dataPackages.value( "Stratified" );
			delete dataPackages.value( "Balanced" );

			if ( mAnalyticsType == "ConfusionMatrixAnalytics" )
			{
				muw::ConfusionMatrixAnalytics* analytics = new muw::ConfusionMatrixAnalytics( trainDP, mAnalyticsUnit );
				mAnalyticsTrain.push_back( analytics );
			}
			else if ( mAnalyticsType == "LossAnalytics" )
			{
				muw::LossAnalytics* analytics = new muw::LossAnalytics( trainDP, mAnalyticsUnit, mAlpha );
				mAnalyticsTrain.push_back( analytics );
			}
		}
	}
	else
	{
		for ( int i = 0; i < mPopulationCount; ++i )
		{
			if ( mAnalyticsType == "ConfusionMatrixAnalytics" )
			{
				muw::ConfusionMatrixAnalytics* analytics = new muw::ConfusionMatrixAnalytics( new muw::DataPackage( trainingDataPackage->featureDB(), trainingDataPackage->labelDB(), trainingDataPackage->labelName() ), mAnalyticsUnit );
				mAnalyticsTrain.push_back( analytics );
			}
			else if ( mAnalyticsType == "LossAnalytics" )
			{
				muw::LossAnalytics* analytics = new muw::LossAnalytics( new muw::DataPackage( trainingDataPackage->featureDB(), mDataPackage->labelDB(), trainingDataPackage->labelName() ), mAnalyticsUnit, mAlpha );
				mAnalyticsTrain.push_back( analytics );
			}
		}
	}

	if ( trainingDataPackage != mDataPackage )
	{
		delete trainingDataPackage;
		trainingDataPackage = nullptr;
	}
}

//-----------------------------------------------------------------------------

void GeneticAlgorithmOptimizer::shuffleDataPackages()
{
	std::shuffle( mAnalyticsTrain.begin(), mAnalyticsTrain.end(), *mGenerator );
}

//-----------------------------------------------------------------------------

void GeneticAlgorithmOptimizer::initialize()
{
	mPopulation.clear();

	std::uniform_int_distribution< unsigned int > rs( 1, mIterationCount * mPopulationCount );
	QVector< unsigned int > randomSeeds;
	for ( int i = 0; i < mPopulationCount; ++i )
	{
		randomSeeds.push_back( rs( *mGenerator ) );
	}

#pragma omp parallel for ordered schedule( dynamic )
	for ( int i = 0; i < mPopulationCount; ++i )
	{
		unsigned int randomSeed = randomSeeds.at( i );
		auto offspring = this->offspring( randomSeed );
		auto fitness = mAnalyticsTrain.at( i )->evaluate( offspring.get() );

#pragma omp critical
		{
			mPopulation.insertMulti( fitness, offspring );
		}
	}

	if ( mIsInternalSubsets ) shuffleDataPackages();
}

//-----------------------------------------------------------------------------

void GeneticAlgorithmOptimizer::evolve()
{
	QMultiMap< double, std::shared_ptr< muw::DEBINN > > newPopulation;

	if ( mGenerationTraining == "Multi" )
	{
		newPopulation = mPopulation;
	}

	std::uniform_int_distribution< unsigned int > rs( 1, mIterationCount * mPopulationCount );
	QVector< unsigned int > randomSeeds;
	for ( int i = 0; i < mPopulationCount; ++i )
	{
		randomSeeds.push_back( rs( *mGenerator ) );
	}

#pragma omp parallel for ordered schedule( dynamic )
	for ( int i = 0; i < mPopulationCount; ++i )
	{
		unsigned int randomSeed = randomSeeds.at( i );
		auto parents   = tournamentSelection();
		auto offspring = this->offspring( parents.first, parents.second, randomSeed );
		auto fitness = mAnalyticsTrain.at( i )->evaluate( offspring.get() );

#pragma omp critical
		{
			newPopulation.insertMulti( fitness, offspring );
		}
	}

	mPopulation.clear();
	for ( int i = 0; i < mPopulationCount; ++i )
	{
		auto fitness = newPopulation.firstKey();
		auto member  = newPopulation.first();

		mPopulation.insertMulti( fitness, member );
		newPopulation.erase( newPopulation.begin() );
	}

	newPopulation.clear();

	if ( mIsInternalSubsets ) shuffleDataPackages();
}

//-----------------------------------------------------------------------------

QVector< double > GeneticAlgorithmOptimizer::result()
{
	qDebug() << "GeneticAlgorithmOptimizer::result() - wrapping up results...";

	if ( mFinalModelSelectionMethod == "LastAlpha" )
	{
		qDebug() << "Returning with last population alpha member.";
		auto fittest = mPopulation.first();
		return fittest->parameters();
	}
	else if ( mFinalModelSelectionMethod == "HistoricalAlpha" )
	{
		qDebug() << "Returning with historical alpha member.";
		auto fittest = mHallOfFame.first();
		return fittest->parameters();
	}
	else if ( mFinalModelSelectionMethod == "HistoricalTest" || mFinalModelSelectionMethod == "LastTest" )
	{
		for (int i = 0; i < mAnalyticsTrain.size(); ++i)
		{
			delete mAnalyticsTrain[ i ]->dataPackage();
			delete mAnalyticsTrain[ i ];
			mAnalyticsTrain[ i ] = nullptr;
		}
		mAnalyticsTrain.clear();

		QMap< QString, double > subsetDefinitions;

		subsetDefinitions[ "Balanced" ] = 0.3;
		subsetDefinitions[ "Stratified" ] = -1.0;

		muw::MultiSubsetFoldGenerator MSFG( mDataPackage, subsetDefinitions, mFinalModelSelectionAnalyticsCount, mIterationCount + 1 );
		MSFG.execute();

		for ( int i = 0; i < mFinalModelSelectionAnalyticsCount; ++i )
		{
			auto subsets = MSFG.fold( i );
			auto currentDataPackage = subsets.value( "Stratified" );
			delete subsets[ "Balanced" ];

			if (mAnalyticsType == "ConfusionMatrixAnalytics")
			{
				muw::ConfusionMatrixAnalytics* analytics = new muw::ConfusionMatrixAnalytics( currentDataPackage, mAnalyticsUnit );
				mAnalyticsTrain.push_back( analytics );
			}
			else if (mAnalyticsType == "LossAnalytics")
			{
				muw::LossAnalytics* analytics = new muw::LossAnalytics( currentDataPackage, mAnalyticsUnit, mAlpha );
				mAnalyticsTrain.push_back( analytics );
			}
		}

		int finalAverageCount = 0;

		std::vector< std::pair< muw::DEBINN*, double > > finalEnsembleCandidates;
		QSet< muw::DEBINN* > uniqueCandidates;

		QList< std::shared_ptr< muw::DEBINN > > population;

		if ( mFinalModelSelectionMethod == "HistoricalTest" )
		{
			qDebug() << "Returning with retesting historical alphas.";
			finalAverageCount = std::min< int >( mEnsembleCount, mHallOfFame.size() );
			population = mHallOfFame.values();
		}
		else if ( mFinalModelSelectionMethod == "LastTest" )
		{
			qDebug() << "Returning with retesting last population members.";
			finalAverageCount = std::min< int >( mEnsembleCount, mPopulation.size() );
			population = mPopulation.values();
		}

		qDebug() << "Ensemble count:" << finalAverageCount;
		
		finalEnsembleCandidates.reserve( population.size() );

		for (int i = 0; i < population.size(); ++i)
		{
			auto member = population[ i ].get();
			if ( !uniqueCandidates.contains( member ) )
			{
				double fitness = 0.0;
				for (int d = 0; d < mAnalyticsTrain.size(); ++d)
				{
					fitness += mAnalyticsTrain.at( d )->evaluate( member );
				}

				fitness /= mAnalyticsTrain.size();
				finalEnsembleCandidates.push_back( std::pair< muw::DEBINN*, double >( member, fitness ) );
				uniqueCandidates.insert( member );
			}
			else
			{
				qDebug() << "WARNING: A member is part of mPopulation multiple times!";
			}
		}

		std::sort( finalEnsembleCandidates.begin(), finalEnsembleCandidates.end(), []( const auto& a, const auto& b )
		{
			return a.second < b.second;
		} );

		QVector< double > ensembleParameters;
		ensembleParameters.resize( population.first()->parameters().size() );
		ensembleParameters.fill( 0.0 );

		for ( int i = 0; i < finalAverageCount; ++i )
		{
			auto currentGenes = finalEnsembleCandidates[ i ].first->parameters();
			for ( int g = 0; g < ensembleParameters.size(); ++g )
			{
				ensembleParameters[ g ] += currentGenes.at( g );
			}
		}

		for ( int g = 0; g < ensembleParameters.size(); ++g )
		{
			ensembleParameters[ g ] /= finalAverageCount;
		}

		qDebug() << "Parameters of final model:";
		qDebug() << ensembleParameters;

		return ensembleParameters;
	}
}

//-----------------------------------------------------------------------------

QPair< std::shared_ptr< muw::DEBINN >, std::shared_ptr< muw::DEBINN > > GeneticAlgorithmOptimizer::tournamentSelection()
{
	auto members = mPopulation.values();
	std::shuffle( members.begin(), members.end(), *mGenerator );

	std::shared_ptr< muw::DEBINN > parent1;
	std::shared_ptr< muw::DEBINN > parent2;

	double fittestMom = DBL_MAX;
	double fittestDad = DBL_MAX;

	for ( int i = 0; i < members.size(); ++i )
	{
		auto member = members.at( i );

		if ( i < members.size() / 2 )  // Parent1
		{
			auto parent1Fitness = mPopulation.key( member );
			if ( parent1Fitness < fittestMom )
			{
				parent1 = member;
				fittestMom = parent1Fitness;
			}
		}
		else  // Parent2
		{
			auto parent2Fitness = mPopulation.key( member );
			if ( parent2Fitness < fittestDad )
			{
				parent2 = member;
				fittestDad = parent2Fitness;
			}
		}
	}

	QPair< std::shared_ptr< muw::DEBINN >, std::shared_ptr< muw::DEBINN > > parents;
	parents.first  = parent1;
	parents.second = parent2;

	return parents;
}

//-----------------------------------------------------------------------------

std::shared_ptr< muw::DEBINN > GeneticAlgorithmOptimizer::offspring( std::shared_ptr< muw::DEBINN > aParent1, std::shared_ptr< muw::DEBINN >  aParent2, unsigned int aRandomSeed )
{
	auto parent1GeneSequence = aParent1->parameters();
	auto parent2GeneSequence = aParent2->parameters();

	QVector< double > offspringGeneSequence;
	offspringGeneSequence.resize( parent1GeneSequence.size() );
	offspringGeneSequence.fill( 0.0 );

	std::uniform_real_distribution< double > dice( 0.0, 1.0 );

	double alignedMutationDelta = mCurrentMutationRate;

	std::mt19937 generator{ aRandomSeed };

	std::uniform_real_distribution< double > mutationRange( -alignedMutationDelta, alignedMutationDelta );

	for ( int i = 0; i < offspringGeneSequence.size(); ++i )
	{
		double selectedGene = 0.0;
		if ( dice( generator ) > 0.5 )
		{
			selectedGene = parent1GeneSequence.at( i );
		}
		else
		{
			selectedGene = parent2GeneSequence.at( i );
		}

		if ( dice( generator ) < mMutationDelta )
		{
			selectedGene += mutationRange( generator );
		}

		offspringGeneSequence[ i ] = selectedGene;
	}

	std::shared_ptr< muw::DEBINN > offspring = std::make_shared< muw::DEBINN >( aParent1->settings() );
	offspring->set( offspringGeneSequence );

	return offspring;
}

//-----------------------------------------------------------------------------

std::shared_ptr< muw::DEBINN > GeneticAlgorithmOptimizer::offspring( unsigned int aRandomSeed )
{
	std::shared_ptr< muw::DEBINN > offspring = std::make_shared< muw::DEBINN >( mAdam->settings() );

	QVector< double > offspringGeneSequence = offspring->parameters();

	std::uniform_real_distribution< double > dice( 0.0, 1.0 );

	double alignedMutationDelta = mCurrentMutationRate;

	std::mt19937 generator{ aRandomSeed };

	std::uniform_real_distribution< double > mutationRange( -alignedMutationDelta, alignedMutationDelta );

	for ( int i = 0; i < offspringGeneSequence.size(); ++i )
	{
		double selectedGene = offspringGeneSequence.at( i );

		if ( dice( generator ) < mMutationDelta )
		{
			selectedGene += mutationRange( generator );
		}

		offspringGeneSequence[ i ] = selectedGene;
	}

	offspring->set( offspringGeneSequence );

	return offspring;
}

//-----------------------------------------------------------------------------

bool GeneticAlgorithmOptimizer::isEarlyStopping( int aCurrentIteration )
{
	double alphaFitness = DBL_MAX;

	if ( !mHallOfFame.isEmpty() )
	{
		alphaFitness = mHallOfFame.firstKey();
	}

	auto population = mPopulation.values();

	if ( aCurrentIteration == 0 )
	{
		while ( true )
		{
			for ( int i = 0; i < mPopulationCount; ++i )
			{
				auto member = population[ i ];
				auto trainFitness = mPopulation.keys().at( i );
				auto validateFitness = mValidationAnalytics->evaluate( member.get() );

				if ( validateFitness < alphaFitness && std::abs( validateFitness - trainFitness ) < mTrainValidateTolerance )
				{
					qDebug() << "New alpha found in iteration" << aCurrentIteration << "- storing alpha in the Hall of Fame with validation fitness" << validateFitness << "( train - validation delta:" << std::abs( validateFitness - trainFitness ) << " )";
					mLastAlphaIteration = aCurrentIteration;
					mHallOfFame.insertMulti( validateFitness, member );
				}
			}

			if ( mHallOfFame.size() < mPopulation.size() )
			{
				mHallOfFame.clear();
				qDebug() << "WARNING: It appears that the Train-Validate Tolerance" << mTrainValidateTolerance << "is too small. Attempting to increase it till reaching a large-enough tolerance margin...";
				mTrainValidateTolerance += 0.05;
			}
			else
			{
				break;
			}
		}
	}
	else
	{
		for ( int i = 0; i < mPopulationCount; ++i )
		{
			auto member = population[ i ];
			auto trainFitness = mPopulation.keys().at( i );
			auto validateFitness = mValidationAnalytics->evaluate( member.get() );

			if ( validateFitness < alphaFitness && std::abs( validateFitness - trainFitness ) < mTrainValidateTolerance )
			{
				qDebug() << "New alpha found in iteration" << aCurrentIteration << "- storing alpha in the Hall of Fame with validation fitness" << validateFitness << "( train - validation delta:" << std::abs( validateFitness - trainFitness ) << " )";
				mLastAlphaIteration = aCurrentIteration;
				mHallOfFame.insertMulti( validateFitness, member );
			}
		}
	}


	while ( mHallOfFame.size() > mPopulationCount )
	{
		mHallOfFame.erase( mHallOfFame.end() - 1 );
	}

	if ( aCurrentIteration - mLastAlphaIteration > mEarlyStoppingMargin )
	{
		if ( !mHallOfFame.isEmpty() )
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	else
	{
		return false;
	}
}

//-----------------------------------------------------------------------------

}
