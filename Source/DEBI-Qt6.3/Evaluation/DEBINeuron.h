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
#include <Evaluation/DEBIStemCell.h>
#include <QVector>
#include <vector>
#include <set>
#include <random>
#include <cfloat>
#include <memory>

namespace muw
{

//-----------------------------------------------------------------------------

/*!
* \brief The DEBINeuron class describes a neuron in the DEBI network.
*
* \details 
* The DEBINeuron class models a neuron in the DEBI network.
* The DEBINeuron has a soma and an axon terminal (axon) node, each represented by x,y,z 3D coordinates. Depending on the neuron type (input, hidden, output), either the soma, or the axon or both are trainable parameters.
* Weights of input neurons (dendrites) are calculated from their distances as of the settings of the DEBINeuron. Note that DEBINeuron refers to its input DEBINeuron axon connections as "dendrites", however, the DEBINN model scheme does not distinguish dendrites as nominal objects.
* A DEBINeuron has two types of axon and soma coordinates: relative coordinates can be trainable parameters (depending on neuron type) and their z-coordinate is always relatie to another spatial object (e.g. axon z coordinates are relative to their soma z coordinates. Soma z coordinates are relative
* to their layer start coordinates). This mechanism allows to train unknown DEBINN parameters with similar value ranges. In contrast to relative coordinates, the aligned soma and axon coordinates have spatially-aligned coordinates that allows them to be positioned correctly in 3D independently of any other spatial objects.
* This mechanism is relied on to calculate weights in an aligned DEBINN model and to render a DEBINN model in 3D.
*/

class Evaluation_API DEBINeuron : public DEBIStemCell
{

public:

	/*!
	\brief NeuronType enum to hold the type of the neuron.
	*/
	enum class NeuronType
	{
		Input = 0,
		Output,
		Hidden
	};

	/*!
	\brief ActionType enum to hold the type of the action potential of the neuron.
	*/
	enum class ActionType
	{
		ReLu = 0,
		LReLu,
		Sigmoid,
		Tanh
	};

	/*!
	\brief DistanceType enum to hold the type of how distances are mapped to weights.
	*/
	enum class DistanceType
	{
		Inverse = 0,
		InverseSquared,
		Gaussian,
		GaussianConst
	};

	/*!
	\brief Geometry enum to hold the type of how spatial geometries are handled in the DEBINN. Currently, DEBI neural networks only handle Euclidean geometries.
	*/
	enum class Geometry
	{
		Euclidean = 0
	};

	/*!
	\brief Simple struct holding x,y,z coordinates.
	*/
	struct Coordinate3D
	{
		Coordinate3D( double aX, double aY, double aZ )
		{
			x = aX;
			y = aY;
			z = aZ;
		}

		double x;
		double y;
		double z;
	};

	/*!
	\brief Copy constructor.
	\param [in] aOther the DEBINeuron to copy in the constructor.
	*/
	DEBINeuron( const DEBINeuron& aOther );

	/*!
	\brief Constructor.
	\param [in] aNeuronType the type of the given neuron as of NeuronType enum.
	\param [in] aCoordinates the list of axon and soma coordinates (x,y,z). In case NeuronType::Hidden, there are two x,y,z coordinates for Soma and Axon respectively. Input neurons have invariant axon coordinates, while Output neurons have invariant axon coordinates.
	\param [in] aActionType the type of the action potential as of ActionType enum.
	\param [in] aDistanceType the type of distance calculation as of DistanceType enum.
	\param [in] aGeometry the type of geometry as of Geometry enum.
	\param [in] aDropOutMinMaxRatio the ratio of dropout across input Dendrites.
	\param [in] aDropoutProbability the probability that dropout occurs across input Dentrites.
	*/
	DEBINeuron( NeuronType aNeuronType, const QVector< double >& aCoordinates, const ActionType& aActionType, const DistanceType& aDistanceType, const Geometry& aGeometry, double aDropOutMinMaxRatio, double aDropoutProbability );

	/*!
	\brief Destructor.
	*/
	~DEBINeuron();

	/*!
	\return  The neuron type as of NeuronType enum.
	*/
	const NeuronType& neuronType() const { return mNeuronType; }

	/*!
	\brief Sets the action type of the neuron.
	\param [in] aActionType the type of the action potential as of ActionType enum.
	*/
	void setActionType( const ActionType& aActionType ) { mActionType = aActionType; }

	/*!
	\return The action type as of ActionType enum.
	*/
	const ActionType& actionType() const { return mActionType; }

	/*!
	\brief Sets the distance type of the neuron.
	\param [in] aDistanceType the type of distance calculation as of DistanceType enum.
	*/
	void setDistanceType( const DistanceType& aDistanceType ) { mDistanceType = aDistanceType; }

	/*!
	\return The distance type as of DistanceType enum.
	*/
	const DistanceType& distanceType() const { return mDistanceType; }

	/*!
	\brief Adds a neuron as a dendrite connection.
	\param [in] aNeuron is another neuron from a previous layer to connect to its axon, modeling it as an input dendrite.
	*/
	void addDendrit( const std::shared_ptr< DEBINeuron >& aNeuron );

	/*!
	\brief Removes a neuron from the list of dendrite connections.
	\param [in] aNeuron is connected neuron to remove from lsit of dendrite connections.
	*/
	void removeDendrit( const std::shared_ptr< DEBINeuron >& aNeuron );

	/*!
	\return The axon coordinates (x,y,z).
	*/
	Coordinate3D& axon() { return mAxon; }

	/*!
	\return The axon coordinates (x,y,z).
	*/
	const Coordinate3D& axon() const { return mAxon; }

	/*!
	\return The soma coordinates (x,y,z).
	*/
	Coordinate3D& soma() { return mSoma; }

	/*!
	\return The soma coordinates (x,y,z).
	*/
	const Coordinate3D& soma() const { return mSoma; }

	/*!
	\return The relative axon coordinates (x,y,z).
	*/
	Coordinate3D& axonRelative() { return mAxonRelative; }

	/*!
	\return The relative axon coordinates (x,y,z).
	*/
	const Coordinate3D& axonRelative() const { return mAxonRelative; }

	/*!
	\return The relative soma coordinates (x,y,z).
	*/
	Coordinate3D& somaRelative() { return mSomaRelative; }

	/*!
	\return The relative soma coordinates (x,y,z).
	*/
	const Coordinate3D& somaRelative() const { return mSomaRelative; }

	/*!
	\return The shift value as of group normalizaiton.
	*/
	const double& shift() const { return mShift; }

	/*!
	\return The shift value as of group normalization.
	*/
	double& shift() { return mShift; }

	/*!
	\return The scale value as of group normalization.
	*/
	const double& scale() const { return mScale; }

	/*!
	\return The scale value as of group normalization.
	*/
	double& scale() { return mScale; }

	/*!
	\return The dropout ratio.
	*/
	const double& dropoutRatio() const { return mDropOutMinMaxRatio; }
	
	/*!
	\return The dropout ratio.
	*/
	double& dropoutRatio() { return mDropOutMinMaxRatio; }

	/*!
	\return The dropout probability.
	*/
	const double& dropoutProbability() const { return mDropoutProbability; }

	/*!
	\return The dropout probability.
	*/
	double& dropoutProbability() { return mDropoutProbability; }

	/*!
	\brief Sets the soma and axon coordinates. In case of Input and Output neurons, only axon and soma coordinates are set respectively. Hidden neurons set both soma and axon coordinates. z-coordinates are maintained to be positive to ensure uni-directionality of the network.
	\param [in] aCoordinates is the vector containing all coordinates of the network to read from.
	\param [in] aStart is the start position in the coordinate vector to start reading from.
	*/
	inline void setCoordinates( const QVector< double > & aCoordinates, int aStart )
	{
		if ( mNeuronType == NeuronType::Input )
		{
			mAxonRelative.x = aCoordinates.at( aStart );
			mAxonRelative.y = aCoordinates.at( aStart + 1 );
			mAxonRelative.z = std::abs( aCoordinates.at( aStart + 2 ) );
		}
		else if ( mNeuronType == NeuronType::Hidden )
		{
			mSomaRelative.x = aCoordinates.at( aStart );
			mSomaRelative.y = aCoordinates.at( aStart + 1 );
			mSomaRelative.z = std::abs( aCoordinates.at( aStart + 2 ) );

			mAxonRelative.x = aCoordinates.at( aStart + 3 );
			mAxonRelative.y = aCoordinates.at( aStart + 4 );
			mAxonRelative.z = std::abs( aCoordinates.at( aStart + 5 ) );
		}
		else if ( mNeuronType == NeuronType::Output )
		{
			mSomaRelative.x = aCoordinates.at( aStart );
			mSomaRelative.y = aCoordinates.at( aStart + 1 );
			mSomaRelative.z = std::abs( aCoordinates.at( aStart + 2 ) );
		}
	}

	/*!
	\brief Returns with the list of neurons that are conencted to the given neuron as input dendrites.
	\return The list of connected neurons registered as dendrites.
	*/
	const std::vector< std::shared_ptr< DEBINeuron > >& dendrites() const { return mDendrites; }

	/*!
	\brief Returns with the list of weights corresponding to the connected dendrites.
	\return The list of weights of connected dendrites.
	*/
	const std::vector< double >& weights() const { return mWeights; }

	/*!
	\brief Calculates the weights of connected dendrites as of DistanceType and Geometry enums.
	*/
	void calculateWeights();

	/*!
	\return The calculated action potential of the neuron.
	*/
	double actionPotential();

	/*!
	\brief Sets the action potnetial of the neuron.
	\param [in] aActionPotential the action potential of the neuron.
	*/
	inline void setActionPotential( double aActionPotential ) { mActionPotential = aActionPotential; mIsValidPotential = true; }

	/*!
	\return The number of trainable parameters of the neuron.
	*/
	const int parameterCount() const { return mParameterCount;  }

	/*!
	\brief Resets the neuron input signal and action potential values.
	*/
	inline void reset()
	{
		mInputSignal      = 0.0;
		mActionPotential  = 0.0;
		mIsValidPotential = false;
	}

	/*!
	\return The length of the neuron within the given network layer as of the sum of its soma nad axon relative z-coordiantes.
	*/
	inline double zLength() { return mSomaRelative.z + mAxonRelative.z; }

	/*!
	\brief Realigns the neuron coordinates within their current layers as of their relative coordinates. In case of Hidden and Output neuron types, aZStart acts as a z-shift coordinate of the given layer.
	\param [in] aZStart the start position of te given Hidden or Output layer. Ignored in case of Input neuron types.
	*/
	inline void realign( double aZStart )
	{
		if ( mNeuronType == NeuronType::Input )
		{
			mAxon.x = mAxonRelative.x;
			mAxon.y = mAxonRelative.y;
			mAxon.z = mSomaRelative.z + mAxonRelative.z;
		}
		else if ( mNeuronType == NeuronType::Hidden )
		{
			mSoma.x = mSomaRelative.x;
			mSoma.y = mSomaRelative.y;
			mSoma.z = mSomaRelative.z + aZStart;

			mAxon.x = mAxonRelative.x;
			mAxon.y = mAxonRelative.y;
			mAxon.z = mAxonRelative.z + mSoma.z;
		}
		else if ( mNeuronType == NeuronType::Output )
		{
			mSoma.x = mSomaRelative.x;
			mSoma.y = mSomaRelative.y;
			mSoma.z = mSomaRelative.z + aZStart;

			mAxon.x = mAxonRelative.x;
			mAxon.y = mAxonRelative.y;
			mAxon.z = mSoma.z + mAxonRelative.z;
		}
	}

	/*!
	\brief Normalizes weights by the input aMean and aStDev values as of z-score standardization.
	\param [in] aMean the mean value of the z-score standardization.
	\param [in] aStDev the standard deviation of the z-score standardization.
	*/
	inline void normalizeWeights( double aMean, double aStDev )
	{
		double devWeightsRec = 1.0 / aStDev;

		for ( int i = 0; i < mWeights.size(); ++i )
		{
			double normalizedWeight = ( mWeights.at( i ) - aMean ) * devWeightsRec;
			mWeights[ i ] = normalizedWeight;
		}
	}

	/*!
	\brief Sets the meand and deviation of the input potentials of connected dendrites.
	\param [in] aMean the mean potential value of connected dendrites.
	\param [in] aStDev the standard deviation potential of tconnected dendrites.
	*/
	inline void setInputMeanDevPotential( double aMean, double aStDev )
	{
		mInputPotentialMean  = aMean;
		mInputPotentialStDevRec = 1.0 / ( aStDev + DBL_EPSILON );
	}

private:

	/*!
	\brief Performs input dendrite and corresponding weight purification as of the dropout ratio and dropout probability.
	*/
	inline void purifyInputs()
	{
		std::uniform_real_distribution< double > range( 0.0, 1.0 );

		if ( mDropOutMinMaxRatio > DBL_EPSILON )
		{
			double minWeight =  DBL_MAX;
			double maxWeight = -DBL_MAX;

			for ( int i = 0; i < mWeights.size(); ++i )
			{
				double currentWeight = mWeights.at( i );
				if ( currentWeight < minWeight ) minWeight = currentWeight;
				if ( currentWeight > maxWeight ) maxWeight = currentWeight;
			}

			double dropoutWeightThreshold = ( maxWeight - minWeight ) * mDropOutMinMaxRatio;
			std::vector< std::shared_ptr< DEBINeuron > > newDendrites;
			std::vector< double >                        newWeights;

			mDendritesSet.clear();

			for ( int i = 0; i < mWeights.size(); ++i )
			{
				double currentWreight = mWeights.at( i );
				if ( currentWreight - minWeight < dropoutWeightThreshold )
				{
					if ( range( *mGenerator ) > mDropoutProbability )
					{
						const auto& dendrite = mDendrites[ i ];
						mDendritesSet.insert( dendrite.get() );

						newDendrites.push_back( mDendrites[ i ] );
						newWeights.push_back( mWeights[ i ] );
					}
				}
				else
				{
					const auto& dendrite = mDendrites[ i ];
					mDendritesSet.insert( dendrite.get() );

					newDendrites.push_back( mDendrites[ i ] );
					newWeights.push_back( mWeights[ i ] );
				}
			}

			mDendrites = newDendrites;
			mWeights   = newWeights;
		}
	}

	/*!
	\brief Maps distances of connected dendrites to weights by inverse quared weighting.
	*/
	inline void calculateInverseSquaredSpatialWeights()
	{
		for ( const auto& dendrite : mDendrites )
		{
			const auto& dendriteAxon   = dendrite->axon();

			double distanceX = mSoma.x - dendriteAxon.x;
			double distanceY = mSoma.y - dendriteAxon.y;
			double distanceZ = mSoma.z - dendriteAxon.z;

			double weight = 9.0 / ( ( distanceX * distanceX ) + ( distanceY * distanceY ) + ( distanceZ * distanceZ ) + DBL_EPSILON );

			mWeights.push_back( weight );
		}
	}

	/*!
	\brief Maps distances of connected dendrites to weights by inverse distance weighting.
	*/
	inline void calculateInverseSpatialWeights()
	{
		for ( const auto& dendrite : mDendrites )
		{
			const auto& dendriteAxon = dendrite->axon();

			double distanceX = mSoma.x - dendriteAxon.x;
			double distanceY = mSoma.y - dendriteAxon.y;
			double distanceZ = mSoma.z - dendriteAxon.z;

			double weight = 3.0 / std::sqrt( ( distanceX * distanceX ) + ( distanceY * distanceY ) + ( distanceZ * distanceZ ) + DBL_EPSILON );

			mWeights.push_back( weight );
		}
	}

	/*!
	\brief Maps distances of connected dendrites to weights by Guassian weighting.
	*/
	inline void calculateGaussianWeights()
	{
		double devDistances = 0.0;

		for ( const auto& dendrite : mDendrites )
		{
			const auto& dendriteAxon = dendrite->axon();

			double distanceX = mSoma.x - dendriteAxon.x;
			double distanceY = mSoma.y - dendriteAxon.y;
			double distanceZ = mSoma.z - dendriteAxon.z;
			devDistances += ( distanceX * distanceX ) + ( distanceY * distanceY ) + ( distanceZ * distanceZ );
		}

		devDistances /= 3.0 * mDendrites.size();
		double recDenominator = 1.0 / ( 2.0 * devDistances );

		for ( int i = 0; i < mDendrites.size(); ++i )
		{
			const auto& dendrite = mDendrites.at( i );
			const auto& dendriteAxon = dendrite->axon();

			double distanceX = mSoma.x - dendriteAxon.x;
			double distanceY = mSoma.y - dendriteAxon.y;
			double distanceZ = mSoma.z - dendriteAxon.z;
			double weight = std::exp( -( ( distanceX * distanceX ) + ( distanceY * distanceY ) + ( distanceZ * distanceZ ) ) * recDenominator );

			mWeights.push_back( weight );
		}
	}

	/*!
	\brief Maps distances of connected dendrites to weights by Guassian weighting where the standard deviation value of the Gaussian function is a constant, allowing the trianing process to grow the network till an optimal Gaussian normalization factor is achieved for distances.
	*/
	inline void calculateGaussianConstWeights()
	{
		double recDenominator = 1.0 / 9.0;

		for ( const auto& dendrite : mDendrites )
		{
			const auto& dendriteAxon = dendrite->axon();

			double distanceX = mSoma.x - dendriteAxon.x;
			double distanceY = mSoma.y - dendriteAxon.y;
			double distanceZ = mSoma.z - dendriteAxon.z;
			double weight = std::exp( -( ( distanceX * distanceX ) + ( distanceY * distanceY ) + ( distanceZ * distanceZ ) ) * recDenominator );

			mWeights.push_back( weight );
		}
	}

	/*!
	\brief Calculates the input signal of the neuron as its input dendrite action potentials, weights as well as the neuron's shift and scale parameters.
	*/
	inline void calculateInputSignal()
	{
		mInputSignal = 0.0;

		for ( int i = 0; i < mDendrites.size(); ++i )
		{
			const auto& dendrite = mDendrites[ i ];
			const auto& weight   = mWeights[ i ];

			mInputSignal += ( mScale * ( ( dendrite->actionPotential() - mInputPotentialMean ) * mInputPotentialStDevRec ) + mShift ) * weight;
		}
	}

	/*!
	\brief Maps the input signal of the neuron to action potential by the ReLu approach.
	*/
	inline void calculateReLuActionPotential()
	{
		mActionPotential = mInputSignal < 0.0 ? 0.0 : mInputSignal;
	}

	/*!
	\brief Maps the input signal of the neuron to action potential by the leaky ReLu approach.
	*/
	inline void calculateLeakyReLuActionPotential()
	{
		mActionPotential = mInputSignal < 0.0 ? 0.01 * mInputSignal : mInputSignal;
	}

	/*!
	\brief Maps the input signal of the neuron to action potential by the Sigmoid approach.
	*/
	inline void calculateSigmoidActionPotential()
	{
		mActionPotential = 1.0 / ( 1.0 + std::exp(- mInputSignal ) );
	}

	/*!
	\brief Maps the input signal of the neuron to action potential by the Tanh approach.
	*/
	inline void calculateTanhActionPotential()
	{
		mActionPotential = ( 2.0 / ( 1.0 + std::exp( -2.0 * mInputSignal ) ) ) - 1.0;
	}

private:

	/*!
	\brief Default constructor (not allowed).
	*/
	DEBINeuron();

private:

	Coordinate3D                                  mSomaRelative;            //!< The relative soma coordinate of the neuron in the given layer.
	Coordinate3D                                  mAxonRelative;            //!< The relative axon coordinate of the neuron in the given layer.
	Coordinate3D                                  mSoma;                    //!< The aligned soma coordinate of the neuron in the given layer.
	Coordinate3D                                  mAxon;                    //!< The aligned axon coordinate of the neuron in the given layer.
	NeuronType                                    mNeuronType;              //!< The type of neuron (input, hidden, output).
	ActionType                                    mActionType;              //!< The type of action potential for activation function calculations.
	DistanceType                                  mDistanceType;            //!< The type of distance handling.
	Geometry                                      mGeometry;                //!< The type of geometry.
	std::vector< std::shared_ptr< DEBINeuron > >  mDendrites;               //!< List of dendrites (connected neurons from previous layers).
	std::set< DEBINeuron* >                       mDendritesSet;            //!< Internal container of dendrites for processing speed optimization.
	std::vector< double >                         mWeights;                 //!< List of calculated weights of dendrites.
	double                                        mInputSignal;             //!< The input signal of the neuron.
	double                                        mActionPotential;         //!< The action potential neuron mapped from the input signal as of an activation function.
	bool                                          mIsValidPotential;        //!< Stores if the action potential of the neuron is already calculated.
	double                                        mDropOutMinMaxRatio;      //!< Ratio of min-max dropout of input dendrites.
	double                                        mDropoutProbability;      //!< Probability that an input dendrite dropout will accur.
	std::mt19937*                                 mGenerator;               //!< Random generator to generate probabilistic values for random calculations (e.g. dropout).
	int                                           mParameterCount;          //!< Number of trainable parameters of the neuron.
	double                                        mScale;                   //!< Scale parameter as of group normalization.
	double                                        mShift;                   //!< Shift parameter as of group normalization.
	double                                        mInputPotentialMean;      //!< Mean of the input potentials of connected dendrites.
	double                                        mInputPotentialStDevRec;  //!< Reciprocal of the standard deviation of the input potentials of connected dendrites.
};

//-----------------------------------------------------------------------------

}
