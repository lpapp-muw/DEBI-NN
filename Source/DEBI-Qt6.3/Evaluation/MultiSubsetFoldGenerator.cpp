/* !
*\file
* This file is part of Evaluation module.
* Member class definitions of MultiSubsetFoldGenerator.
*
* \remarks
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#include <Evaluation/MultiSubsetFoldGenerator.h>
#include <Evaluation/TabularDataFilter.h>
#include <FileIo/TabularDataFileIo.h>
#include <DataRepresentation/Types.h>
#include <QDebug>
#include <algorithm>
#include <cfloat>
#include <random>

extern bool isRandomSeed;

namespace muw
{

//-----------------------------------------------------------------------------

MultiSubsetFoldGenerator::MultiSubsetFoldGenerator( muw::DataPackage* aDataPackage, QMap< QString, double > aSubsetDefinitions, int aFoldCount, unsigned int aRandomSeed )
:
	mDataPackage( aDataPackage ),
	mSubsetDefinitions( aSubsetDefinitions ),
	mFoldCount( aFoldCount ),
	mAlignedSubsetDefinitions( aSubsetDefinitions ),
	mAlignedSubsetCounts(),
	mFoldDefinitions(),
	mGenerator( nullptr ),
	mIsValid( true )
{
	if ( isRandomSeed )
	{
		std::random_device rd{};
		mGenerator = new std::mt19937{ rd() };
	}
	else
	{
		mGenerator = new std::mt19937{ aRandomSeed };
	}
}

//-----------------------------------------------------------------------------

MultiSubsetFoldGenerator::MultiSubsetFoldGenerator( muw::DataPackage* aDataPackage, QMap< QString, double > aSubsetDefinitions, QString aFoldConfigurationFilePath )
:
	mDataPackage( aDataPackage ),
	mSubsetDefinitions( aSubsetDefinitions ),
	mFoldCount(0),
	mAlignedSubsetDefinitions(),
	mAlignedSubsetCounts(),
	mFoldDefinitions(),
	mGenerator( nullptr ),
	mIsValid( true )
{
	muw::TabularDataFileIo configIO;
	configIO.load( aFoldConfigurationFilePath + "/FDB-Folds.csv", mFoldDefinitions );

	mFoldCount = mFoldDefinitions.columnCount();

	if ( isRandomSeed )
	{
		std::random_device rd{};
		mGenerator = new std::mt19937{ rd() };
	}
	else
	{
		mGenerator = new std::mt19937{ 123 };
	}
}

//-----------------------------------------------------------------------------

void MultiSubsetFoldGenerator::execute()
{
	muw::TabularDataFilter filter;
	auto labelOutcomes = mDataPackage->labelOutcomes();
	QMap< QString, QStringList > keysOfLabelOutcomes;
	auto sampleKeys = mDataPackage->sampleKeys();
	std::sort( sampleKeys.begin(), sampleKeys.end() );
	qDebug() << "Label outcomes: " << labelOutcomes;

	// Calculate the number of label outcomes of the label and identify the minority outcome.
	QString minimumLabelOutcomeKey;
	int minimumLabelOutcomeValue = INT_MAX;

	QMap< QString, int > labelOutcomeCounts;
	for ( auto labelOutcome : labelOutcomes )
	{
		auto LDBReduced = filter.subTableByKeys( mDataPackage->labelDB(), sampleKeys );
		auto keysOfLabelOutcome = filter.keysByLabelGroup( LDBReduced, mDataPackage->labelIndex(), labelOutcome );
		keysOfLabelOutcomes.insert( labelOutcome, keysOfLabelOutcome );
		qDebug() << "Count of label outcome " << labelOutcome << "is" << keysOfLabelOutcome.count();
		labelOutcomeCounts.insert( labelOutcome, keysOfLabelOutcome.count() );
		if ( keysOfLabelOutcome.count() < minimumLabelOutcomeValue )
		{
			minimumLabelOutcomeValue = keysOfLabelOutcome.count();
			minimumLabelOutcomeKey = labelOutcome;
		}
	}

	// Identify the flexible subset.
	QString flexibleSubsetKey;
	for ( auto subset : mSubsetDefinitions.keys() )
	{
		if ( mSubsetDefinitions.value( subset ) < 0.0 )
		{
			flexibleSubsetKey = subset;
		}
	}

	// Correct the aligned subset definitions.
	for ( auto subset : mSubsetDefinitions.keys() )
	{
		if ( subset == flexibleSubsetKey ) continue;
		double subsetOfLabelOutcomeCount = mSubsetDefinitions.value( subset ) * minimumLabelOutcomeValue;
		if ( subsetOfLabelOutcomeCount < 1 )
		{
			mAlignedSubsetDefinitions[ subset ] = ( mSubsetDefinitions.value( subset ) * ( 1.0 / subsetOfLabelOutcomeCount ) ) + DBL_EPSILON;
		}
	}

	// Calculate the sum of subsamples
	int sum = 0;
	for ( auto subset : mAlignedSubsetDefinitions.keys() )
	{
		if ( subset == flexibleSubsetKey ) continue;
		int subsetCount = mAlignedSubsetDefinitions.value( subset ) * minimumLabelOutcomeValue;
		qDebug() << "Subset" << subset << "count: " << subsetCount;
		mAlignedSubsetCounts[ subset ] = subsetCount;
		sum += subsetCount;
	}

	// Valid state?
	if ( sum >= minimumLabelOutcomeValue )
	{
		qDebug() << "ERROR: Not enough samples to generate all subsamples! Sum " << sum << "vs minimum subset count: " << minimumLabelOutcomeValue;
		mIsValid = false;
	}
	else
	{
		qDebug() << "Original subset ratios: " << mSubsetDefinitions;
		qDebug() << "Aligned subset ratios: " << mAlignedSubsetDefinitions;
		qDebug() << "Aligned subset counts:";
		for ( auto labelOutcome : labelOutcomeCounts.keys() )
		{
			auto labelOutcomeCount = labelOutcomeCounts.value( labelOutcome );
			for ( auto subset : mAlignedSubsetDefinitions.keys() )
			{
				if ( subset == flexibleSubsetKey )
				{
					qDebug() << subset << "-" << labelOutcome << ": " << labelOutcomeCount - sum;
				}
				else
				{
					auto subsetCount = mAlignedSubsetCounts.value( subset );
					qDebug() << subset << "-" << labelOutcome << ": " << subsetCount;
				}
			}
		}
	}

	//For each fold: generate random subsets and make sure that the new subset is not generated already.
	QVector< QString > shuffledKeyHistory;
	QVector< QVector< QString > > shuffledKeySubsets;
	int i = 0;
	bool isFinal = false;

	// Generate fold configurations.
	while ( !isFinal )
	{
		if ( i >= mFoldCount )
		{
			isFinal = true;
			break;
		}

		// Make sure that the shuffled keys have not been generated before.
		int duplicateCount = 0;
		bool isUniqueFound = false;
		while ( !isUniqueFound )
		{
			// Shuffle the keys of the label outcome
			for ( auto labelOutcome : labelOutcomes )
			{
				std::shuffle( keysOfLabelOutcomes[ labelOutcome ].begin(), keysOfLabelOutcomes[ labelOutcome ].end(), *mGenerator );
			}

			QVector< QString > shuffledKeySubset;
			shuffledKeySubset.resize( sampleKeys.size() );
			shuffledKeySubset.fill( flexibleSubsetKey );

			for ( auto labelOutcome : labelOutcomes )
			{
				int index = 0;
				int summedSubsetCount = 0;
				for ( auto subset : mAlignedSubsetCounts.keys() )
				{
					if ( subset == flexibleSubsetKey ) continue;
					int subsetCount = mAlignedSubsetCounts.value( subset );
					summedSubsetCount += subsetCount;

					while ( index < summedSubsetCount )
					{
						auto sampleKey = keysOfLabelOutcomes[ labelOutcome ].at( index );
						shuffledKeySubset[ sampleKeys.indexOf( sampleKey ) ] = subset;
						++index;
					}
				}
			}

			QString shuffledKeyIndices = "SK";
			auto subsetKeys = mAlignedSubsetDefinitions.keys();
			for ( int k = 0; k < shuffledKeySubset.size(); ++k )
			{
				auto subset = shuffledKeySubset.at( k );
				shuffledKeyIndices += "-" + QString::number( subsetKeys.indexOf( subset ) );
			}

			// The new shuffle is unique?
			if ( shuffledKeyHistory.contains( shuffledKeyIndices ) )
			{
				qDebug() << "OOPS, already existing shuffle generated...";
				isUniqueFound = false;
				++duplicateCount;
				if ( duplicateCount > mFoldCount )
				{
					qDebug() << "It is not possible to make new folds. Stopping at fold" << i;
					mFoldCount = i;
					isFinal = true;
					break;
				}
			}
			else
			{
				shuffledKeyHistory.push_back( shuffledKeyIndices );
				shuffledKeySubsets.push_back( shuffledKeySubset );
				duplicateCount = 0;
				isUniqueFound = true;
				++i;
			}
		}
	}

	// Initiate the header and fill in the fold configuration table.
	QStringList header;
	QVariantList featureVector;

	for ( int i = 0; i < mFoldCount; ++i )
	{
		header.push_back( "Fold-" + QString::number( i ) );
		featureVector.push_back( flexibleSubsetKey );
	}
	mFoldDefinitions.setHeader( header );

	for ( int i = 0; i < sampleKeys.size(); ++i )
	{
		mFoldDefinitions.insert( sampleKeys.at( i ), featureVector );
	}

	for ( int i = 0; i < mFoldCount; ++i )
	{
		auto shuffledKeySubset = shuffledKeySubsets.at( i );
		for ( int s = 0; s < sampleKeys.size(); ++s )
		{
			auto sampleKey = sampleKeys.at( s );
			mFoldDefinitions[ sampleKey ][ i ] = shuffledKeySubset.at( s );
		}
	}

	//// Save fold definitions to path.
	//muw::TabularDataFileIo configIO;
	//configIO.save( mFoldConfigurationFilePath + "/FoldConfiguration.csv", mFoldDefinitions );
}

//-----------------------------------------------------------------------------

QMap< QString, muw::DataPackage* > MultiSubsetFoldGenerator::fold( int aFold )
{
	QMap< QString, muw::DataPackage* > generatedFold;
	if ( mIsValid )
	{
		muw::TabularDataFilter filter;
		auto sampleKeys = mDataPackage->sampleKeys();

		// Generate DataPackages for each subset and return with them.
		auto subsets = mSubsetDefinitions.keys();
		for ( auto subset : subsets )
		{
			QStringList subsetKeys;
			for ( auto sampleKey : sampleKeys )
			{
				if ( mFoldDefinitions.valueAt( sampleKey, aFold ) == subset )
				{
					subsetKeys.push_back( sampleKey );
				}
			}
			auto FDB = filter.subTableByKeys( mDataPackage->featureDB(), subsetKeys );
			auto LDB = filter.subTableByKeys( mDataPackage->labelDB(), subsetKeys );
			QString labelName = mDataPackage->labelName();

			muw::DataPackage* DP = new muw::DataPackage( FDB, LDB, labelName );
			generatedFold.insert( subset, DP );
		}
	}

	return generatedFold;
}

//-----------------------------------------------------------------------------

}