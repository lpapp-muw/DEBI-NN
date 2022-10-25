/*!
* \file
* Member class definitions of DEBINeron. This file is part of the Evaluation module.
*
* \remarks
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#include <Evaluation/DEBINeuron.h>
#include <DataRepresentation/Types.h>

extern bool isRandomSeed;

namespace muw
{

//-----------------------------------------------------------------------------

DEBINeuron::DEBINeuron( const DEBINeuron& aOther )
:
	DEBIStemCell( "DEBINeuron" ),
	mSomaRelative( aOther.mSomaRelative ),
	mAxonRelative( aOther.mAxonRelative ),
	mSoma( aOther.mSoma ),
	mAxon( aOther.mAxon ),
	mNeuronType( aOther.mNeuronType ),
	mActionType( aOther.mActionType ),
	mDistanceType( aOther.mDistanceType ),
	mGeometry( aOther.mGeometry ),
	mDendrites( aOther.mDendrites ),
	mWeights( aOther.mWeights ),
	mInputSignal( aOther.mInputSignal ),
	mActionPotential( aOther.mActionPotential ),
	mIsValidPotential( false ),
	mDropOutMinMaxRatio( aOther.mDropOutMinMaxRatio ),
	mDropoutProbability( aOther.mDropoutProbability ),
	mGenerator( aOther.mGenerator ),
	mParameterCount( aOther.mParameterCount ),
	mScale( aOther.mScale ),
	mShift( aOther.mShift ),
	mInputPotentialMean( aOther.mInputPotentialMean ),
	mInputPotentialStDevRec( aOther.mInputPotentialStDevRec )
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

DEBINeuron::DEBINeuron( NeuronType aNeuronType, const QVector< double >& aCoordinates, const ActionType& aActionType, const DistanceType& aDistanceType, const Geometry& aGeometry, double aDropOutMinMaxRatio, double aDropoutProbability )
:
	DEBIStemCell( "DEBINeuron" ),
	mSomaRelative( Coordinate3D( 0.0, 0.0, 0.0 ) ),
	mAxonRelative( Coordinate3D( 0.0, 0.0, 0.0 ) ),
	mSoma( Coordinate3D( 0.0, 0.0, 0.0 ) ),
	mAxon( Coordinate3D( 0.0, 0.0, 0.0 ) ),
	mNeuronType( aNeuronType ),
	mActionType( aActionType ),
	mDistanceType( aDistanceType ),
	mGeometry( aGeometry ),
	mDendrites(),
	mWeights(),
	mInputSignal( 0.0 ),
	mActionPotential( -DBL_MAX ),
	mIsValidPotential( false ),
	mDropOutMinMaxRatio( aDropOutMinMaxRatio ),
	mDropoutProbability( aDropoutProbability ),
	mGenerator( nullptr ),
	mParameterCount( 0 ),
	mScale( 1.0 ),
	mShift( 0.0 ),
	mInputPotentialMean( 0.0 ),
	mInputPotentialStDevRec( 1.0 )
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

	if ( mNeuronType == NeuronType::Input )
	{
		mParameterCount = 3;

		mSomaRelative.x = 0.0;
		mSomaRelative.y = 0.0;
		mSomaRelative.z = 0.0;

		mSoma.x = mSomaRelative.x;
		mSoma.y = mSomaRelative.y;
		mSoma.z = mSomaRelative.z;
			 
		mAxonRelative.x = aCoordinates.at( 0 );
		mAxonRelative.y = aCoordinates.at( 1 );
		mAxonRelative.z = std::abs( aCoordinates.at( 2 ) );
	}
	else if ( mNeuronType == NeuronType::Hidden )
	{
		mParameterCount = 6;

		mSomaRelative.x = aCoordinates.at( 0 );
		mSomaRelative.y = aCoordinates.at( 1 );
		mSomaRelative.z = std::abs( aCoordinates.at( 2 ) );
			 
		mAxonRelative.x = aCoordinates.at( 3 );
		mAxonRelative.y = aCoordinates.at( 4 );
		mAxonRelative.z = std::abs( aCoordinates.at( 5 ) );
	}
	else if ( mNeuronType == NeuronType::Output )
	{
		mParameterCount = 3;

		mSomaRelative.x = aCoordinates.at( 0 );
		mSomaRelative.y = aCoordinates.at( 1 );
		mSomaRelative.z = std::abs( aCoordinates.at( 2 ) );
			 
		mAxonRelative.x = 0.0;
		mAxonRelative.y = 0.0;
		mAxonRelative.z = 0.5;
	}
}

//-----------------------------------------------------------------------------

DEBINeuron::~DEBINeuron()
{
	mDendrites.clear();
	mWeights.clear();
	delete mGenerator;
	mGenerator = nullptr;
}

//-----------------------------------------------------------------------------

void DEBINeuron::addDendrit( const std::shared_ptr< DEBINeuron >& aNeuron )
{
	if ( mDendritesSet.find( aNeuron.get() ) == mDendritesSet.end() )
	{
		mDendrites.push_back( aNeuron );
		mDendritesSet.insert( aNeuron.get() );
	}
}

//-----------------------------------------------------------------------------

void DEBINeuron::removeDendrit( const std::shared_ptr< DEBINeuron >& aNeuron )
{
	std::remove( mDendrites.begin(), mDendrites.end(), aNeuron );
	mDendritesSet.erase( aNeuron.get() );
}

//-----------------------------------------------------------------------------

void DEBINeuron::calculateWeights()
{
	mWeights.clear();

	if ( mDistanceType == DistanceType::InverseSquared && mGeometry == Geometry::Euclidean )
	{
		calculateInverseSquaredSpatialWeights();
	}
	else if ( mDistanceType == DistanceType::Inverse && mGeometry == Geometry::Euclidean )
	{
		calculateInverseSpatialWeights();
	}
	else if ( mDistanceType == DistanceType::Gaussian && mGeometry == Geometry::Euclidean )
	{
		calculateGaussianWeights();
	}
	else if ( mDistanceType == DistanceType::GaussianConst && mGeometry == Geometry::Euclidean )
	{
		calculateGaussianConstWeights();
	}

	purifyInputs();
}

//-----------------------------------------------------------------------------

double DEBINeuron::actionPotential()
{
	if ( !mIsValidPotential )
	{
		calculateInputSignal();

		if ( mActionType == ActionType::ReLu )
		{
			calculateReLuActionPotential();
		}
		else if ( mActionType == ActionType::LReLu )
		{
			calculateLeakyReLuActionPotential();
		}
		else if ( mActionType == ActionType::Sigmoid )
		{
			calculateSigmoidActionPotential();
		}
		else if ( mActionType == ActionType::Tanh )
		{
			calculateTanhActionPotential();
		}

		mIsValidPotential = true;
	}
	
	return mActionPotential;
}

//-----------------------------------------------------------------------------

}
