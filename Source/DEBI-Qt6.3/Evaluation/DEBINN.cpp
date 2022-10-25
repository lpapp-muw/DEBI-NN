/*!
* \file
* Member class definitions of DEBINN. This file is part of the Evaluation module.
*
* \remarks
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#include <Evaluation/DEBINN.h>
#include <DataRepresentation/Types.h>
#include <QDebug>
#include <QFile>
#include <QDataStream>
#include <QSet>
#include <omp.h>
#include <stdint.h>
#include <random>

extern bool isRandomSeed;

namespace muw
{

//-----------------------------------------------------------------------------

DEBINN::DEBINN( QVariantMap aSettings )
:
	muw::AbstractModel( nullptr ),
	mSettings( aSettings ),
	mLayerCounts(),
	mActionTypes(),
	mDistanceTypes(),
	mGeometries(),
	mHiddenGroupCounts(),
	mLayerGroups(),
	mLayerDeltas(),
	mDropoutRatios(),
	mDropoutProbabilities(),
	mIsRandomInitialize(),
	mDEBINeurons(),
	mZRanges(),
	mPredictionLabels(),
	mGenerator( nullptr ),
	mIsRealigned( false ),
	mIsLayerDeltasOptimize( false ),
	mIsDropoutRatiosOptimize( false ),
	mIsDropoutProbabilitiesOptimize( false ),
	mIsWeightsStandardize( false ),
	mIsGroupNormalize( false ),
	mIsShiftScaleOptimize( false ),
	mGlobalMinWeight( DBL_MAX ),
	mGlobalMaxWeight( -DBL_MAX )
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

	setup();
	initialize();
}

//-----------------------------------------------------------------------------

DEBINN::DEBINN( const DEBINN& aOther )
:
	muw::AbstractModel(              nullptr ),
	mSettings(                       aOther.mSettings ),
	mLayerCounts(                    aOther.mLayerCounts ),
	mActionTypes(                    aOther.mActionTypes ),
	mDistanceTypes(                  aOther.mDistanceTypes ),
	mGeometries(                     aOther.mGeometries ),
	mHiddenGroupCounts(              aOther.mHiddenGroupCounts ),
	mLayerGroups(                    aOther.mLayerGroups ),
	mLayerDeltas(                    aOther.mLayerDeltas ),
	mDropoutRatios(                  aOther.mDropoutRatios ),
	mDropoutProbabilities(           aOther.mDropoutProbabilities ),
	mIsRandomInitialize(             aOther.mIsRandomInitialize ),
	mDEBINeurons(                    aOther.mDEBINeurons ),
	mZRanges(                        aOther.mZRanges ),
	mPredictionLabels(               aOther.mPredictionLabels ),
	mGenerator(                      nullptr ),
	mIsRealigned(                    aOther.mIsRealigned ),
	mIsLayerDeltasOptimize(          aOther.mIsLayerDeltasOptimize ),
	mIsDropoutRatiosOptimize(        aOther.mIsDropoutRatiosOptimize ),
	mIsDropoutProbabilitiesOptimize( aOther.mIsDropoutProbabilitiesOptimize ),
	mIsWeightsStandardize(           aOther.mIsWeightsStandardize ),
	mIsGroupNormalize(               aOther.mIsGroupNormalize ),
	mIsShiftScaleOptimize(           aOther.mIsShiftScaleOptimize ),
	mGlobalMinWeight(                aOther.mGlobalMinWeight ),
	mGlobalMaxWeight(                aOther.mGlobalMaxWeight )
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
}

//-----------------------------------------------------------------------------

void DEBINN::setup()
{
	// Input layer settings.
	int     inputNeuronCount     = mSettings.value( "Model/Layers/Input/NeuronCount" ).toInt();
	QString inputActionType      = mSettings.value( "Model/Layers/Input/ActionType" ).toString();
	QString inputDistanceType    = mSettings.value( "Model/Layers/Input/DistanceType" ).toString();
	QString inputGeometry = mSettings.value( "Model/Layers/Input/Geometry" ).toString();
	double  inputLayerDelta      = mSettings.value( "Model/Layers/Input/LayerDelta" ).toDouble();

	mLayerCounts.push_back( inputNeuronCount );
	if ( inputActionType      == "LReLu" )   mActionTypes.push_back( DEBINeuron::ActionType::LReLu );
	else if ( inputActionType == "ReLu" )    mActionTypes.push_back( DEBINeuron::ActionType::ReLu );
	else if ( inputActionType == "Sigmoid" ) mActionTypes.push_back( DEBINeuron::ActionType::Sigmoid );
	else if ( inputActionType == "Tanh" )    mActionTypes.push_back( DEBINeuron::ActionType::Tanh );

	if ( inputDistanceType      == "InverseSquared" ) mDistanceTypes.push_back( DEBINeuron::DistanceType::InverseSquared );
	else if ( inputDistanceType == "Inverse" )        mDistanceTypes.push_back( DEBINeuron::DistanceType::Inverse );
	else if ( inputDistanceType == "Gaussian" )       mDistanceTypes.push_back( DEBINeuron::DistanceType::Gaussian );
	else if ( inputDistanceType == "GaussianConst" )  mDistanceTypes.push_back( DEBINeuron::DistanceType::GaussianConst );

	if ( inputGeometry == "Euclidean" ) mGeometries.push_back( DEBINeuron::Geometry::Euclidean );

	mLayerDeltas.push_back( inputLayerDelta );
	mDropoutRatios.push_back( 0.0 );
	mDropoutProbabilities.push_back( 0.0 );

	// Hidden layer settings.
	QStringList hiddenNeuronCounts         = mSettings.value( "Model/Layers/Hidden/NeuronCount" ).toStringList();
	QStringList hiddenActionTypes          = mSettings.value( "Model/Layers/Hidden/ActionType" ).toStringList();
	QStringList hiddenDistanceTypes        = mSettings.value( "Model/Layers/Hidden/DistanceType" ).toStringList();
	QStringList hiddenGeometrys     = mSettings.value( "Model/Layers/Hidden/Geometry" ).toStringList();
	QStringList hiddenGroupCounts          = mSettings.value( "Model/Layers/Hidden/GroupCount" ).toStringList();
	QStringList hiddenLayerDeltas          = mSettings.value( "Model/Layers/Hidden/LayerDelta" ).toStringList();
	QStringList hiddenDropoutRatios        = mSettings.value( "Model/Layers/Hidden/DropoutRatio" ).toStringList();
	QStringList hiddenDropoutProbabilities = mSettings.value( "Model/Layers/Hidden/DropoutProbability" ).toStringList();

	for ( int h = 0; h < hiddenNeuronCounts.size(); ++h )
	{
		mLayerCounts.push_back( hiddenNeuronCounts.at( h ).toInt() );
		
		if (      hiddenActionTypes.at( h ) == "LReLu" )   mActionTypes.push_back( DEBINeuron::ActionType::LReLu );
		else if ( hiddenActionTypes.at( h ) == "ReLu" )    mActionTypes.push_back( DEBINeuron::ActionType::ReLu );
		else if ( hiddenActionTypes.at( h ) == "Sigmoid" ) mActionTypes.push_back( DEBINeuron::ActionType::Sigmoid );
		else if ( hiddenActionTypes.at( h ) == "ReLu" )    mActionTypes.push_back( DEBINeuron::ActionType::Tanh );

		if (      hiddenDistanceTypes.at( h ) == "InverseSquared" ) mDistanceTypes.push_back( DEBINeuron::DistanceType::InverseSquared );
		else if ( hiddenDistanceTypes.at( h ) == "Inverse" )        mDistanceTypes.push_back( DEBINeuron::DistanceType::Inverse );
		else if ( hiddenDistanceTypes.at( h ) == "Gaussian" )       mDistanceTypes.push_back( DEBINeuron::DistanceType::Gaussian );
		else if ( hiddenDistanceTypes.at( h ) == "GaussianConst" )  mDistanceTypes.push_back( DEBINeuron::DistanceType::GaussianConst );

		if ( hiddenGeometrys.at( h ) == "Euclidean" ) mGeometries.push_back( DEBINeuron::Geometry::Euclidean );

		mHiddenGroupCounts.push_back(    hiddenGroupCounts.at( h ).toInt() );
		mLayerDeltas.push_back(          hiddenLayerDeltas.at( h ).toDouble() );
		mDropoutRatios.push_back(        hiddenDropoutRatios.at( h ).toDouble() );
		mDropoutProbabilities.push_back( hiddenDropoutProbabilities.at( h ).toDouble() );
	}

	// Output layer settings.
	int     outputNeuronCount        = mSettings.value( "Model/Layers/Output/NeuronCount" ).toInt();
	QString outputActionType         = mSettings.value( "Model/Layers/Output/ActionType" ).toString();
	QString outputDistanceType       = mSettings.value( "Model/Layers/Output/DistanceType" ).toString();
	QString outputGeometry           = mSettings.value( "Model/Layers/Output/Geometry" ).toString();
	double  outputLayerDelta         = mSettings.value( "Model/Layers/Output/LayerDelta" ).toDouble();
	double  outputDropoutRatio       = mSettings.value( "Model/Layers/Output/DropoutRatio" ).toDouble();
	double  outputDropoutProbability = mSettings.value( "Model/Layers/Output/DropoutProbability" ).toDouble();

	mLayerCounts.push_back( outputNeuronCount );
	if (      outputActionType == "LReLu" )   mActionTypes.push_back( DEBINeuron::ActionType::LReLu );
	else if ( outputActionType == "ReLu" )    mActionTypes.push_back( DEBINeuron::ActionType::ReLu );
	else if ( outputActionType == "Sigmoid" ) mActionTypes.push_back( DEBINeuron::ActionType::Sigmoid );
	else if ( outputActionType == "ReLu" )    mActionTypes.push_back( DEBINeuron::ActionType::Tanh );

	if (      outputDistanceType == "InverseSquared" ) mDistanceTypes.push_back( DEBINeuron::DistanceType::InverseSquared );
	else if ( outputDistanceType == "Inverse" )        mDistanceTypes.push_back( DEBINeuron::DistanceType::Inverse );
	else if ( outputDistanceType == "Gaussian" )       mDistanceTypes.push_back( DEBINeuron::DistanceType::Gaussian );
	else if ( outputDistanceType == "GaussianConst" )  mDistanceTypes.push_back( DEBINeuron::DistanceType::GaussianConst );

	if ( outputGeometry == "Euclidean" ) mGeometries.push_back( DEBINeuron::Geometry::Euclidean );

	mLayerDeltas.push_back( outputLayerDelta );
	mDropoutRatios.push_back( outputDropoutRatio );
	mDropoutProbabilities.push_back( outputDropoutProbability );

	// Global settings.
	mPredictionLabels                    = mSettings.value( "Data/PredictionLabels" ).toStringList();
	QString randomSpatialInitialize      = mSettings.value( "Model/RandomSpatialInitialize" ).toString();
	QString layerDeltasOptimize          = mSettings.value( "Model/LayerDeltasOptimize" ).toString();
	QString dropoutRatiosOptimize        = mSettings.value( "Model/DropoutRatiosOptimize" ).toString();
	QString dropoutProbabilitiesOptimize = mSettings.value( "Model/DropoutProbabilitiesOptimize" ).toString();
	QString isWeightsStandardize         = mSettings.value( "Model/WeightsStandardize" ).toString();
	QString isGroupNormalize             = mSettings.value( "Model/GroupsNormalize" ).toString();
	QString isShiftScaleOptimize         = mSettings.value( "Model/ShiftScaleOptimize" ).toString();
	
	mIsRandomInitialize             = randomSpatialInitialize      == "true" ? true : false;
	mIsLayerDeltasOptimize          = layerDeltasOptimize          == "true" ? true : false;
	mIsDropoutRatiosOptimize        = dropoutRatiosOptimize        == "true" ? true : false;
	mIsDropoutProbabilitiesOptimize = dropoutProbabilitiesOptimize == "true" ? true : false;
	mIsWeightsStandardize           = isWeightsStandardize         == "true" ? true : false;
	mIsGroupNormalize               = isGroupNormalize             == "true" ? true : false;
	mIsShiftScaleOptimize           = isShiftScaleOptimize         == "true" ? true : false;
}

//-----------------------------------------------------------------------------

void DEBINN::initialize()
{
	// Groups configuration.
	for ( int i = 0; i < mLayerCounts.size(); ++i )
	{
		QVector< NeuronGroup > groups;

		if ( i == 0 )  // Input layer.
		{
			NeuronGroup inputNeuronGroup;
			inputNeuronGroup.scale = 1.0;
			inputNeuronGroup.shift = 0.0;

			groups.push_back( inputNeuronGroup );
			mLayerGroups.push_back( groups );
		}
		else if ( i == mLayerCounts.size() - 1 )  // Output layer.
		{
			for ( int j = 0; j < mLayerCounts.at( i ); ++j )
			{
				NeuronGroup outputNeuronGroup;
				outputNeuronGroup.neuronIndices.push_back( j );
				outputNeuronGroup.scale = 1.0;
				outputNeuronGroup.shift = 0.0;
				groups.push_back( outputNeuronGroup );
			}
			mLayerGroups.push_back( groups );
		}
		else  // Hidden layer.
		{
			int neuronIndex = 0;
			int currentGroupCount = mHiddenGroupCounts.at( i - 1 );
			int groupSize = mLayerCounts.at( i ) / currentGroupCount;
			int groupIndex = 0;
			while ( true )
			{
				int startGroup = neuronIndex;
				int endGroup   = std::min( mLayerCounts.at( i ), startGroup + groupSize );
				int remnant    = mLayerCounts.at( i ) - endGroup;
				if ( remnant < endGroup - startGroup )
				{
					endGroup = mLayerCounts.at( i );
				}

				NeuronGroup hiddenNeuronGroup;
				hiddenNeuronGroup.scale = 1.0;
				hiddenNeuronGroup.shift = 0.0;

				for ( int j = startGroup; j < endGroup; ++j )
				{
					hiddenNeuronGroup.neuronIndices.push_back( j );
				}
				groups.push_back( hiddenNeuronGroup );
				
				neuronIndex = endGroup;
				++groupIndex;

				if ( neuronIndex >= mLayerCounts.at( i ) ) break;
			}

			mLayerGroups.push_back( groups );
		}
	}

	// Neurons configuration.
	
	std::uniform_real_distribution< double > coordinate( -1.0, 1.0 );

	for ( int layer = 0; layer < mLayerCounts.size(); ++layer )
	{
		QVector< std::shared_ptr< DEBINeuron > > DEBINeurons;
		int neuronCount = mLayerCounts.value( layer );
		for ( int neuron = 0; neuron < neuronCount; ++neuron )
		{
			QVector< double > coordinates;

			if ( layer == 0 )  // Input layer neuron.
			{
				if ( mIsRandomInitialize )
				{
					coordinates.push_back( coordinate( *mGenerator ) );
					coordinates.push_back( coordinate( *mGenerator ) );
					coordinates.push_back( coordinate( *mGenerator ) );
				}
				else
				{
					coordinates.push_back( 0.0 );
					coordinates.push_back( 0.0 );
					coordinates.push_back( 0.0 );
				}

				DEBINeuron::ActionType actionType           = mActionTypes.at( layer );
				DEBINeuron::DistanceType distanceType       = mDistanceTypes.at( layer );
				DEBINeuron::Geometry Geometry = mGeometries.at( layer );
				double dropoutRatio                            = mDropoutRatios.value( layer );
				double dropoutProbability                      = mDropoutProbabilities.value( layer );
				DEBINeurons.push_back( std::make_shared< DEBINeuron >( DEBINeuron::NeuronType::Input, coordinates, actionType, distanceType, Geometry, dropoutRatio, dropoutProbability ) );
			}
			else if ( layer == mLayerCounts.size() - 1 )  // Output layer neuron.
			{
				if ( mIsRandomInitialize )
				{
					coordinates.push_back( coordinate( *mGenerator ) );
					coordinates.push_back( coordinate( *mGenerator ) );
					coordinates.push_back( coordinate( *mGenerator ) );
				}
				else
				{
					coordinates.push_back( 0.0 );
					coordinates.push_back( 0.0 );
					coordinates.push_back( 0.0 );
				}

				DEBINeuron::ActionType actionType           = mActionTypes.at( layer );
				DEBINeuron::DistanceType distanceType       = mDistanceTypes.at( layer );
				DEBINeuron::Geometry Geometry = mGeometries.at( layer );
				double dropoutRatio                            = mDropoutRatios.value( layer );
				double dropoutProbability                      = mDropoutProbabilities.value( layer );
				DEBINeurons.push_back( std::make_shared< DEBINeuron >( DEBINeuron::NeuronType::Output, coordinates, actionType, distanceType, Geometry, dropoutRatio, dropoutProbability ) );
			}
			else  // Hidden layer neuron.
			{
				if ( mIsRandomInitialize )
				{
					coordinates.push_back( coordinate( *mGenerator ) );
					coordinates.push_back( coordinate( *mGenerator ) );
					coordinates.push_back( coordinate( *mGenerator ) );

					coordinates.push_back( coordinate( *mGenerator ) );
					coordinates.push_back( coordinate( *mGenerator ) );
					coordinates.push_back( coordinate( *mGenerator ) );
				}
				else
				{
					coordinates.push_back( 0.0 );
					coordinates.push_back( 0.0 );
					coordinates.push_back( 0.0 );

					coordinates.push_back( 0.0 );
					coordinates.push_back( 0.0 );
					coordinates.push_back( 0.0 );
				}

				DEBINeuron::ActionType actionType           = mActionTypes.at( layer );
				DEBINeuron::DistanceType distanceType       = mDistanceTypes.at( layer );
				DEBINeuron::Geometry Geometry = mGeometries.at( layer );
				double dropoutRatio                            = mDropoutRatios.value( layer );
				double dropoutProbability                      = mDropoutProbabilities.value( layer );

				DEBINeurons.push_back( std::make_shared< DEBINeuron >( DEBINeuron::NeuronType::Hidden, coordinates, actionType, distanceType, Geometry, dropoutRatio, dropoutProbability ) );
			}
		}

		mDEBINeurons.push_back( DEBINeurons );
	}

	integrate();
}

//-----------------------------------------------------------------------------

DEBINN::~DEBINN()
{
	mLayerCounts.clear();
	mActionTypes.clear();
	mDistanceTypes.clear();
	mGeometries.clear();
	mHiddenGroupCounts.clear();
	mLayerGroups.clear();
	mLayerDeltas.clear();
	mDropoutRatios.clear();
	mDropoutProbabilities.clear();
	mDEBINeurons.clear();
	mZRanges.clear();
	mPredictionLabels.clear();
	mGenerator = nullptr;
}

//-----------------------------------------------------------------------------

int DEBINN::inputCount()
{
	int parameterCount = 0;

	int layerCount = mDEBINeurons.size();
	for ( int layer = 0; layer < layerCount; ++layer )
	{
		if ( mIsShiftScaleOptimize )
		{
			const auto& currentGroups = mLayerGroups.at( layer );
			for ( const auto& currentGroup : currentGroups )
			{
				if ( layer == 0 ) continue;
				parameterCount += 2;
			}
		}

		if ( mIsLayerDeltasOptimize )
		{
			++parameterCount;
		}

		if ( mIsDropoutRatiosOptimize )
		{
			if ( layer == 0 ) continue;
			++parameterCount;
		}

		if ( mIsDropoutProbabilitiesOptimize )
		{
			if ( layer == 0 ) continue;
			++parameterCount;
		}
	}

	for ( auto DEBINeurons : mDEBINeurons )
	{
		for ( auto DEBINeuron : DEBINeurons )
		{
			parameterCount += DEBINeuron->parameterCount();
		}
	}

	return parameterCount;
}

//-----------------------------------------------------------------------------

QVector< double > DEBINN::parameters()
{
	QVector< double > parameters;

	int layerCount = mDEBINeurons.size();
	for (int layer = 0; layer < layerCount; ++layer)
	{
		//auto currentLayer  = mDEBINeurons.at( layer );

		if ( mIsShiftScaleOptimize )
		{
			auto currentGroups = mLayerGroups.at( layer );
			for ( auto currentGroup : currentGroups )
			{
				if ( layer == 0 ) continue;

				auto scale = currentGroup.scale;
				auto shift = currentGroup.shift;

				parameters.append( scale );
				parameters.append( shift );
			}
		}

		if ( mIsLayerDeltasOptimize )
		{
			auto layerDelta = mLayerDeltas.at( layer );
			parameters.append( layerDelta );
		}
		
		if ( mIsDropoutRatiosOptimize )
		{
			if (layer == 0) continue;
			auto dropoutRatio = mDropoutRatios.at(layer);
			parameters.append( dropoutRatio );
		}

		if ( mIsDropoutProbabilitiesOptimize )
		{
			if ( layer == 0 ) continue;
			auto dropoutProbability = mDropoutProbabilities.at( layer );
			parameters.append( dropoutProbability );
		}
	}

	for ( auto DEBINeurons : mDEBINeurons )
	{
		for ( auto DEBINeuron : DEBINeurons )
		{
			if ( DEBINeuron->neuronType() == DEBINeuron::NeuronType::Input )
			{
				const auto& axon = DEBINeuron->axonRelative();
				parameters.append( { axon.x, axon.y, axon.z } );
			}
			else if ( DEBINeuron->neuronType() == DEBINeuron::NeuronType::Hidden )
			{
				const auto& soma = DEBINeuron->somaRelative();
				const auto& axon = DEBINeuron->axonRelative();

				parameters.append( { soma.x, soma.y, soma.z, axon.x, axon.y, axon.z } );
			}
			else if ( DEBINeuron->neuronType() == DEBINeuron::NeuronType::Output )
			{
				const auto& soma = DEBINeuron->somaRelative();

				parameters.append( { soma.x, soma.y, soma.z } );
			}
		}
	}

	return parameters;
}

//-----------------------------------------------------------------------------

void DEBINN::set( const QVector< double >& aParameters )
{
	int globalIndex = 0;

	int layerCount = mDEBINeurons.size();
	for ( int layerID = 0; layerID < layerCount; ++layerID )
	{
		if ( mIsShiftScaleOptimize )
		{
			const auto& currentLayer  = mDEBINeurons.at( layerID );
			const auto& currentGroups = mLayerGroups.at( layerID );

			for ( int groupID = 0; groupID < currentGroups.size(); ++groupID )
			{
				if ( layerID == 0 ) continue;

				auto currentScale = aParameters.at( globalIndex );
				auto currentShift = aParameters.at( globalIndex + 1 );
				globalIndex += 2;

				mLayerGroups[ layerID ][ groupID ].scale = currentScale;
				mLayerGroups[ layerID ][ groupID ].shift = currentShift;

				const QVector< int >& currentNeuronIndices = mLayerGroups.at( layerID ).at( groupID ).neuronIndices;
				for ( int i = 0; i < currentNeuronIndices.size(); ++i )
				{
					auto currentNeuronIndex = currentNeuronIndices.at( i );
					auto currentDEBINeuron = currentLayer.at( currentNeuronIndex );
					currentDEBINeuron->shift() = currentShift;
					currentDEBINeuron->scale() = currentScale;
				}
			}
		}

		if ( mIsLayerDeltasOptimize )
		{
			auto currentLayerDelta = aParameters.at( globalIndex );
			mLayerDeltas[ layerID ] = currentLayerDelta;
			++globalIndex;
		}
		
		if ( mIsDropoutRatiosOptimize )
		{
			if (layerID == 0) continue;
			auto currentDropoutRatio = std::abs( aParameters.at( globalIndex ) );
			mDropoutRatios[layerID] = currentDropoutRatio;
			++globalIndex;
		}

		if ( mIsDropoutProbabilitiesOptimize )
		{
			if ( layerID == 0 ) continue;
			auto currentDropoutProbability = std::abs( aParameters.at( globalIndex ) );
			mDropoutProbabilities[ layerID ] = currentDropoutProbability;
			++globalIndex;
		}
	}

	for ( int layer = 0; layer < layerCount; ++layer )
	{
		const auto& currentLayer = mDEBINeurons.at( layer );
		for ( int i = 0; i < currentLayer.size(); ++i )
		{
			const auto& currentDEBINeuron = currentLayer.at( i );
			int parameterCount = currentDEBINeuron->parameterCount();

			currentDEBINeuron->setCoordinates( aParameters, globalIndex );
			currentDEBINeuron->dropoutRatio()       = mDropoutRatios.value( layer );
			currentDEBINeuron->dropoutProbability() = mDropoutProbabilities.value( layer );
			globalIndex += parameterCount;
		}
	}

	realign();

	mIsRealigned = true;
}

//-----------------------------------------------------------------------------

void DEBINN::realign()
{
	// Realign the network spatially.
	mZRanges.clear();
	int layerCount = mDEBINeurons.size();
	for ( int layer = 0; layer < layerCount; ++layer )
	{

		auto currentLayerDelta = mLayerDeltas.at( layer );
		auto currentLayer      = mDEBINeurons.at( layer );
		double maxZLength = 0.0;
		for ( int i = 0; i < currentLayer.size(); ++i )
		{
			const auto& currentDEBINeuron = currentLayer.at( i );
			double currentZLength     = currentDEBINeuron->zLength();
			if ( currentZLength > maxZLength ) maxZLength = currentZLength;
		}

		if ( layer == 0 )
		{
			mZRanges.push_back( QPair< double, double >( 0.0, maxZLength ) );
		}
		else
		{
			const auto& previousRange = mZRanges.at( layer - 1 );
			double previousThickness = previousRange.second - previousRange.first;

			double currentZStart = ( previousThickness * currentLayerDelta ) + previousRange.second;
			double currentZEnd = currentZStart + maxZLength;

			mZRanges.push_back( QPair< double, double >( currentZStart, currentZEnd ) );
		}

		for ( int i = 0; i < currentLayer.size(); ++i )
		{
			const auto& currentDEBINeuron = currentLayer.at( i );
			double currentZStart = mZRanges.at( layer ).first;
			currentDEBINeuron->realign( currentZStart );
		}
	}

	// Calculate initial weights per neuron.
	for ( int layer = 1; layer < layerCount; ++layer )
	{
		const auto& currentLayer = mDEBINeurons.at( layer );

#pragma omp parallel for ordered schedule( dynamic )
		for ( int i = 0; i < currentLayer.size(); ++i )
		{
			const auto& currentDEBINeuron = currentLayer.at( i );
			currentDEBINeuron->calculateWeights();
		}
	}

	// Weight standardization
	mGlobalMinWeight =  DBL_MAX;
	mGlobalMaxWeight = -DBL_MAX;

	if ( mIsWeightsStandardize )
	{
		for ( int layer = 1; layer < layerCount; ++layer )
		{
			auto currentLayer = mDEBINeurons.at( layer );

			std::vector< double > weights;
			weights.reserve( currentLayer.size() );

			for ( int i = 0; i < currentLayer.size(); ++i )
			{
				auto currentDEBINeuron = currentLayer.at( i );
				const auto& neuronWeights = currentDEBINeuron->weights();
				weights.insert( weights.end(), neuronWeights.begin(), neuronWeights.end() );
			}

			QVector< double > normalizedWeights;
			auto meanDevWeights = meanStDevOfVector( weights );

			for ( int i = 0; i < currentLayer.size(); ++i )
			{
				const auto& currentDEBINeuron = currentLayer.at( i );
				currentDEBINeuron->normalizeWeights( meanDevWeights.first, meanDevWeights.second );
				const auto& standardizedWeights = currentDEBINeuron->weights();
				for ( auto weight : standardizedWeights )
				{
					if ( weight > mGlobalMaxWeight ) mGlobalMaxWeight = weight;
					if ( weight < mGlobalMinWeight ) mGlobalMinWeight = weight;
				}
			}
		}
	}
	else
	{
		for ( int layer = 1; layer < layerCount; ++layer )
		{
			auto currentLayer = mDEBINeurons.at( layer );

			for ( int i = 0; i < currentLayer.size(); ++i )
			{
				const auto& currentDEBINeuron = currentLayer.at( i );
				const auto& weights = currentDEBINeuron->weights();
				for ( auto weight : weights )
				{
					if ( weight > mGlobalMaxWeight ) mGlobalMaxWeight = weight;
					if ( weight < mGlobalMinWeight ) mGlobalMinWeight = weight;
				}
			}
		}
	}

	mIsRealigned = true;
}

//-----------------------------------------------------------------------------

std::vector< double > DEBINN::allWeights()
{
	std::vector< double > weights;

	int layerCount = mDEBINeurons.size();
	for ( int layer = 1; layer < layerCount; ++layer )
	{
		const auto& currentLayer = mDEBINeurons.at( layer );

		
		for ( int i = 0; i < currentLayer.size(); ++i )
		{
			const auto& currentDEBINeuron = currentLayer.at( i );
			const auto& neuronWeights = currentDEBINeuron->weights();
			weights.insert( weights.end(), neuronWeights.begin(), neuronWeights.end() );
		}
	}

	return weights;
}

//-----------------------------------------------------------------------------

void DEBINN::integrate()
{
	int layerCount = mDEBINeurons.size();

	for ( int layer = 1; layer < layerCount; ++layer )
	{
		auto previousLayer = mDEBINeurons.at( layer - 1 );
		auto currentLayer  = mDEBINeurons.at( layer );

		for ( auto currentDEBINeuron : currentLayer )
		{
			for ( auto previousDEBINeuron : previousLayer )
			{
				currentDEBINeuron->addDendrit( previousDEBINeuron );
			}
		}
	}
}

//-----------------------------------------------------------------------------

QVariant DEBINN::evaluate( const QVector< double >& aFeatureVector )
{
	if ( !mIsRealigned ) realign();

	// Reset action potentials and result.
	resetActionPotentials();

	QMap< QString, QVariant > result;
	for ( const auto& predictionLabel : qAsConst( mPredictionLabels ) )
	{
		result[ predictionLabel ] = 0.0;
	}

	// Compatible input?
	int layerCount  = mDEBINeurons.size();
	const auto& inputLayer = mDEBINeurons.value( 0 );

	if ( inputLayer.size() != aFeatureVector.size() )
	{
		qDebug() << "DEBINN::execute - ERROR: Input size and input layer size mismatch:" << aFeatureVector.size(); inputLayer.size();
		return QVariant::fromValue( -1 );
	}

	// Set up the input layer.
	for ( int i = 0; i < aFeatureVector.size(); ++i )
	{
		const auto& inputDEBINeuron = inputLayer.at( i );
		inputDEBINeuron->setActionPotential( aFeatureVector.at( i ) );
	}

	std::vector< double > inputs;

	// Execute the hidden layers.
	for ( int layer = 1; layer < layerCount; ++layer )
	{
		const auto& currentLayer = mDEBINeurons.at( layer );

		// Group normalization.
		const auto& currentGroups = mLayerGroups.at( layer );

		for ( const auto& currentGroup : currentGroups )
		{
			QSet< muw::DEBINeuron* > logInputNeurons;

			inputs.clear();

			for ( auto neuronIndex : qAsConst( currentGroup.neuronIndices ) )
			{
				const auto& currentDEBINeuron = currentLayer.at( neuronIndex );
				const auto& dendrites = currentDEBINeuron->dendrites();

				for ( const auto& dendrite : dendrites )
				{
					if ( logInputNeurons.contains( dendrite.get() ) ) continue;
					double potential = dendrite->actionPotential();
					inputs.push_back( potential );

					logInputNeurons.insert( dendrite.get() );
				}
			}

			QPair< double, double > meanDevGroupPotentials;

			if ( mIsGroupNormalize )
			{
				meanDevGroupPotentials = meanStDevOfVector( inputs );
			}
			else
			{
				meanDevGroupPotentials.first  = 0.0;
				meanDevGroupPotentials.second = 1.0;
			}

			for ( auto neuronIndex : qAsConst( currentGroup.neuronIndices ) )
			{
				const auto& currentDEBINeuron = currentLayer.at( neuronIndex );
				currentDEBINeuron->setInputMeanDevPotential( meanDevGroupPotentials.first, meanDevGroupPotentials.second );
				currentDEBINeuron->actionPotential();
			}
		}
	}

	// Read out the output layer values.
	const auto& outputLayer = mDEBINeurons.at( layerCount - 1 );

	std::vector< double > normalizedScores;
	normalizedScores.reserve( outputLayer.size() );

	double sumAbsScores = 0.0;

	for ( int i = 0; i < outputLayer.size(); ++i )
	{
		const auto& currentDEBINeuron   = outputLayer.at( i );
		const auto& currentActionPotential = currentDEBINeuron->actionPotential();
		normalizedScores.push_back( currentActionPotential );
		sumAbsScores += std::abs( currentActionPotential );
	}

	double sumSoftMax = 0.0;
	std::vector< double > softMaxes;
	softMaxes.reserve( outputLayer.size() );

	for ( int i = 0; i < outputLayer.size(); ++i )
	{
		//auto currentDEBINeuron = outputLayer.at( i );
		//auto currentActionPotential = currentDEBINeuron->actionPotential();

		double currentScore   = normalizedScores[ i ] / ( sumAbsScores + DBL_EPSILON );
		double currentSoftMax = std::exp( currentScore );
		softMaxes.push_back( currentSoftMax );
		sumSoftMax += currentSoftMax;
	}

	for ( int i = 0; i < outputLayer.size(); ++i )
	{
		result[ mPredictionLabels.at( i ) ] = softMaxes[ i ] / ( sumSoftMax + DBL_EPSILON );
	}

	return QVariant::fromValue( result );
}

//-----------------------------------------------------------------------------

void DEBINN::resetActionPotentials()
{
	int layerCount = mDEBINeurons.size();
	for ( int layer = 0; layer < layerCount; ++layer )
	{
		auto currentLayer = mDEBINeurons.at( layer );
		for ( auto currentDEBINeuron : currentLayer )
		{
			currentDEBINeuron->reset();
		}
	}
}

//-----------------------------------------------------------------------------

QVector< double > DEBINN::calculateBoundingBox()
{
	QVector< double > result;
	result.resize( 6 );
	result[ 0 ] =  DBL_MAX;
	result[ 1 ] =  DBL_MAX;
	result[ 2 ] =  DBL_MAX;
	result[ 3 ] = -DBL_MAX;
	result[ 4 ] = -DBL_MAX;
	result[ 5 ] = -DBL_MAX;

	int layerCount = mDEBINeurons.size();

	for ( int layer = 0; layer < layerCount; ++layer )
	{
		const auto& currentLayer = mDEBINeurons.at( layer );
		for ( const auto& currentDEBINeuron : currentLayer )
		{
			auto axon = currentDEBINeuron->axon();
			auto soma = currentDEBINeuron->soma();

			if ( axon.x < result.at( 0 ) ) result[ 0 ] = axon.x;
			if ( axon.y < result.at( 1 ) ) result[ 1 ] = axon.y;
			if ( axon.z < result.at( 2 ) ) result[ 2 ] = axon.z;
			if ( soma.x < result.at( 0 ) ) result[ 0 ] = soma.x;
			if ( soma.y < result.at( 1 ) ) result[ 1 ] = soma.y;
			if ( soma.z < result.at( 2 ) ) result[ 2 ] = soma.z;

			if ( axon.x > result.at( 3 ) ) result[ 3 ] = axon.x;
			if ( axon.y > result.at( 4 ) ) result[ 4 ] = axon.y;
			if ( axon.z > result.at( 5 ) ) result[ 5 ] = axon.z;
			if ( soma.x > result.at( 3 ) ) result[ 3 ] = soma.x;
			if ( soma.y > result.at( 4 ) ) result[ 4 ] = soma.y;
			if ( soma.z > result.at( 5 ) ) result[ 5 ] = soma.z;
		}
	}

	return result;
}

//-----------------------------------------------------------------------------

QVector< double > DEBINN::center()
{
	QVector< double > center;
	center.resize( 3 );
	center.fill( 0.0 );

	auto boundingBox = calculateBoundingBox();

	center[ 0 ] = ( ( boundingBox.at( 3 ) - boundingBox.at( 0 ) ) * 0.5 ) + boundingBox.at( 0 );
	center[ 1 ] = ( ( boundingBox.at( 4 ) - boundingBox.at( 1 ) ) * 0.5 ) + boundingBox.at( 1 );
	center[ 2 ] = ( ( boundingBox.at( 5 ) - boundingBox.at( 2 ) ) * 0.5 ) + boundingBox.at( 2 );

	return center;
}

//-----------------------------------------------------------------------------

double DEBINN::l1Norm()
{
	int layerCount = mDEBINeurons.size();
	double l1Norm = 0.0;

	for ( int layer = 1; layer < layerCount; ++layer )
	{
		const auto& currentLayer = mDEBINeurons.at( layer );
		for ( const auto& currentDEBINeuron : currentLayer )
		{
			const auto& weights = currentDEBINeuron->weights();

			double l1NormOfNeuron = 0.0;
			for ( auto weight : weights )
			{
				l1NormOfNeuron += std::abs( weight );
			}

			l1Norm += l1NormOfNeuron / weights.size();
		}
	}

	return l1Norm;
}

//-----------------------------------------------------------------------------

double DEBINN::l2Norm()
{
	int layerCount = mDEBINeurons.size();
	double l2Norm = 0.0;

	for ( int layer = 1; layer < layerCount; ++layer )
	{
		const auto& currentLayer = mDEBINeurons.at( layer );
		for ( const auto& currentDEBINeuron : currentLayer )
		{
			const auto& weights = currentDEBINeuron->weights();

			double l2NormOfNeuron = 0.0;
			for ( auto weight : weights )
			{
				l2NormOfNeuron += weight * weight;
			}

			l2Norm += l2NormOfNeuron / weights.size();
		}
	}

	return l2Norm;
}

//-----------------------------------------------------------------------------

void DEBINN::save( QString aFilePath )
{
	QFile outFile( aFilePath );
	if ( outFile.open( QIODevice::WriteOnly ) )
	{
		QDataStream out( &outFile );
		out.setVersion(QDataStream::Qt_4_5);
		out.setFloatingPointPrecision( QDataStream::FloatingPointPrecision::DoublePrecision );

		out << mLayerCounts;

		QVector< int > actionTypes;
		for ( auto actionType : mActionTypes )
		{
			actionTypes.push_back( int( actionType ) );
		}
		out << actionTypes;

		QVector< int > distanceTypes;
		for ( auto distanceType : mDistanceTypes )
		{
			distanceTypes.push_back( int( distanceType ) );
		}
		out << distanceTypes;

		QVector< int > Geometrys;
		for ( auto Geometry : mGeometries )
		{
			Geometrys.push_back( int( Geometry ) );
		}
		out << Geometrys;

		out << mHiddenGroupCounts;

		out << mLayerGroups;

		out << mLayerDeltas;
		out << mDropoutRatios;
		out << mDropoutProbabilities;

		out << this->parameters();

		out << mIsRandomInitialize;
		out << mZRanges;
		out << mPredictionLabels;
		out << mIsRealigned;
		out << mIsLayerDeltasOptimize;
		out << mIsDropoutRatiosOptimize;
		out << mIsDropoutProbabilitiesOptimize;
		out << mIsWeightsStandardize;
		out << mIsGroupNormalize;
		out << mIsShiftScaleOptimize;
		out << mGlobalMinWeight;
		out << mGlobalMaxWeight;

	}
	else
	{
		qDebug() << "Error: file cannot be open to write" << aFilePath;
	}
}

//-----------------------------------------------------------------------------

void DEBINN::load( QString aFilePath )
{
	QFile inFile( aFilePath );
	if ( inFile.open( QIODevice::ReadOnly ) )
	{
		QDataStream in( &inFile );
		in.setVersion(QDataStream::Qt_4_5);
		in.setFloatingPointPrecision( QDataStream::FloatingPointPrecision::DoublePrecision );

		in >> mLayerCounts;
		
		mActionTypes.clear();
		QVector< int > actionTypes;
		in >> actionTypes;
		for ( auto actionType : actionTypes )
		{
			mActionTypes.push_back( DEBINeuron::ActionType( actionType ) );
		}

		mDistanceTypes.clear();
		QVector< int > distanceTypes;
		in >> distanceTypes;
		for ( auto distanceType : distanceTypes )
		{
			mDistanceTypes.push_back( DEBINeuron::DistanceType( distanceType ) );
		}

		mGeometries.clear();
		QVector< int > Geometrys;
		in >> Geometrys;
		for ( auto Geometry : Geometrys )
		{
			mGeometries.push_back( DEBINeuron::Geometry( Geometry ) );
		}

		in >> mHiddenGroupCounts;
		in >> mLayerGroups;
		in >> mLayerDeltas;
		in >> mDropoutRatios;
		in >> mDropoutProbabilities;

		QVector< double > parameters;
		in >> parameters;

		in >> mIsRandomInitialize;
		in >> mZRanges;
		in >> mPredictionLabels;
		in >> mIsRealigned;
		in >> mIsLayerDeltasOptimize;
		in >> mIsDropoutRatiosOptimize;
		in >> mIsDropoutProbabilitiesOptimize;
		in >> mIsWeightsStandardize;
		in >> mIsGroupNormalize;
		in >> mIsShiftScaleOptimize;
		in >> mGlobalMinWeight;
		in >> mGlobalMaxWeight;

		this->set( parameters );
	}
	else
	{
		qDebug() << "Error: file cannot be open to read" << aFilePath;
	}
}

//-----------------------------------------------------------------------------

}

QDataStream& operator<<( QDataStream& aOut, const muw::DEBINN::NeuronGroup& aNeuronGroup )
{
	aOut << aNeuronGroup.neuronIndices;
	aOut << aNeuronGroup.scale;
	aOut << aNeuronGroup.shift;

	return aOut;
}

//-----------------------------------------------------------------------------

QDataStream& operator>>( QDataStream& aIn, muw::DEBINN::NeuronGroup& aNeuronGroup )
{
	aIn >> aNeuronGroup.neuronIndices;
	aIn >> aNeuronGroup.scale;
	aIn >> aNeuronGroup.shift;

	return aIn;
}

//-----------------------------------------------------------------------------
