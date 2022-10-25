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
#include <Evaluation/DEBINeuron.h>
#include <Evaluation/AbstractModel.h>
#include <QMap>
#include <QVariant>


namespace muw
{

//-----------------------------------------------------------------------------

/*!
* \brief The DEBINN class manifests the mechanisms of a DEBI neural network. 
*
* \details
* A DEBINN can be initialized either initially by basic input parameters as provided by its QSettings, or by loading up an already trained, stored DEBINN model.
* The DEBINN class is able to model a neural network having input, one or more hidden and an output neuron layer, where each layer contains DEBINeuron instances.
* Training a DEBINN is performed throughout optimizing the relative spatial coordinates of its DEBINeuron instances. A relative coordinate is composed of an x,y,z coordinate triplet. The z-coordinate of the given Axon of a DEBINeuron
* is relative to the given Soma of that DEBINeuron. Similarly, the z-coordinate of the Soma is relative to the spatial start position of the given neural layer it contains the DEBINeuron. A DEBINN layer start-end in x,y and z directions 
* is equal with the smallest and largest spatial coordinate of all included DEBINeurons within.
* While the x and y coordinates can grow in both - and + directions, the z-coordinates are currently modified to their absolute value to provide uni-directionality for the network.
* Once a set of trainable parameter is provided to the given network, it spatially realigns itself by calculating final 3D coordinates for all its spatial elements from their respective relative coordinate system.
* An aligned DEBINN then undergoes weight calculations based on layer X Soma and layer X-1 Axons that are connected (note that the current implementation considers fully-connected networks).
* Once weights are calculated by a distance-to-weight mapping function, a DEBINN acts identically to a conventional informational NN.
* The DEBINN also incldues group normalization mechanisms that - based on its settings - can be activated/deactivated. When activated, the network also considers neuron groups and also handles a shift-scale parameter pair for each group.
*/
class Evaluation_API DEBINN : public muw::AbstractModel
{

public:

	/*!
	\brief NeuronGroup struct to hold the list of neuron indices of a given layer together with their shift and scale parameters for group normalization.
	*/
	struct NeuronGroup
	{
		QVector< int > neuronIndices;
		double scale;
		double shift;
	};

	/*!
	\brief Copy constructor.
	\param [in] aSettings the settings that contains parameters to initialize a DEBI neural network
	*/
	DEBINN( QVariantMap aSettings );

	/*!
	\brief Copy constructor.
	\param [in] aOther the DEBI neural network to copy in the constructor.
	*/
	DEBINN( const DEBINN& aOther );
	
	/*!
	\brief Destructor.
	*/
	~DEBINN();

	/*!
	\brief Returns with all trainable parameters of the network.
	\return the QVector of parameters that are trainable.
	*/
	QVector< double > parameters();

	/*!
	\brief Returns with all neurons of all layers of the network.
	\return the QVector of layers and their neurons in embedded QVectors.
	*/
	QVector< QVector< std::shared_ptr< DEBINeuron > > > DEBINeurons() { return mDEBINeurons; }

	/*!
	\brief Sets the trainable parameters for the network.
	\param [in] aParameters the vector of trainable parameters to set.
	*/
	void set( const QVector< double >& aParameters );

	/*!
	\brief Realigns the network spatially as of its spatial parameters.
	*/
	void realign();

	/*!
	\brief Provides a prediction for the input feature vector.
	\param [in] aFeatureVector the vector containing the input features to evalaute and predict from.
	\return The Qvariant containing the prediction result.
	*/
	QVariant evaluate( const QVector< double >& aFeatureVector );

	/*!
	\brief Returns with the number of trainable parameters of the network.
	\return The number of trainable parameters of the network.
	*/
	int inputCount();

	/*!
	\brief Calculates the spatial bounding box enclosing the network.
	\return The start x,y,z and end x,y,z coordinates of the bounding box of the network.
	*/
	QVector< double > calculateBoundingBox();

	/*!
	\brief Calculates the center coordinate of the neural network.
	\return The  x,y,z coordinate of the center of the neural network.
	*/
	QVector< double > center();

	/*!
	\brief Returns with the z-ranges (minimum and maximum z-coordinate) for each layer of the neural network.
	\return The z-ranges per layer of the neural network.
	*/
	QVector< QPair< double, double > > zRanges() { return mZRanges; }

	/*!
	\return True if the network is realigned spatially. False otherwise.
	*/
	bool isValid() { return mIsRealigned; }

	/*!
	\return The L1 norm (non-normalized by sample count) of the network. Currently not used.
	*/
	double l1Norm();

	/*!
	\return The L2 norm (non-normalized by sample count) of the network. Currently not used.
	*/
	double l2Norm();

	/*!
	\return The number of neurons per layer in the network.
	*/
	const QVector< int >& layerConfiguration() const { return mLayerCounts; }

	/*!
	\return The labels the network is able to predict. Note that the current implementation is only able to perform supervised learning and classification.
	*/
	const QStringList& predictionLabels() const { return mPredictionLabels; }

	/*!
	\return The settings of the network.
	*/
	const QVariantMap& settings() const { return mSettings; }

	/*!
	\return The smallest weight in the network.
	*/
	double globalMinWeight() { return mGlobalMinWeight; }

	/*!
	\return The largest weight in the network.
	*/
	double globalMaxWeight() { return mGlobalMaxWeight; }

	/*!
	\brief Saves the network to the file provided as input in binary format.
	\param [in] aFilePath the file path of the file in which the network is saved.
	*/
	void save( QString aFilePath );

	/*!
	\brief Loads the network from the file provided as input.
	\param [in] aFilePath the file path of the file from which the network is loaded.
	*/
	void load( QString aFilePath );

	/*!
	\return All weights of the network.
	*/
	std::vector< double > allWeights();

private:

	/*!
	\brief Reads out all setting parameter from the setting file provided in the contructor.
	*/
	void setup();

	/*!
	\brief Initializes the network with random spatial coordinates as of the settings provided in the contructor. 
	*/
	void initialize();

	/*!
	\brief Connects DEBI neurons across consecutive layers to be fully-connected.
	*/
	void integrate();

	/*!
	\brief Default contructor cannot be used.
	*/
	DEBINN();

	/*!
	\brief Resets all previously claculated acton potentials in the network.
	*/
	void resetActionPotentials();

	/*!
	\brief Calculates the mean and standard deviation of values provided as input.
	\param [in] aVector the vector containing floating point numbers.
	\return The mean and standard deviation pair of the input.
	*/
	template < typename VectorType >
	inline QPair< double, double > meanStDevOfVector( const VectorType& aVector )
	{
		double mean = 0.0;
		double recCount = 1.0 / ( aVector.size() + DBL_EPSILON );

		for ( int i = 0; i < aVector.size(); ++i )
		{
			mean += aVector[ i ];
		}
		mean *= recCount;

		double stDev = 0.0;
		for ( int i = 0; i < aVector.size(); ++i )
		{
			stDev += ( aVector[ i ] - mean ) * ( aVector[ i ] - mean );
		}
		stDev *= recCount;
		stDev = std::sqrt( stDev );

		return QPair< double, double >( mean, stDev );
	}

private:

	QVariantMap                                          mSettings;						   //!< The settings container to set up the network upon creation.
	QVector< int >                                       mLayerCounts;					   //!< The number of neurons per layer.
	QVector< DEBINeuron::ActionType >                    mActionTypes;					   //!< The types of action potentials per layer.
	QVector< DEBINeuron::DistanceType >                  mDistanceTypes;				   //!< The distance-to-weight mapper types per layer.
	QVector< DEBINeuron::Geometry >                      mGeometries;					   //!< The spatial gemoetries per layer.
	QVector< int >                                       mHiddenGroupCounts;			   //!< The number of groups in the hidden layer.
	QVector < QVector< NeuronGroup > >                   mLayerGroups;					   //!< The groups of neurons per layer.
	QVector< double >                                    mLayerDeltas;					   //!< The ratio of layer overlaps in between consecutive layers.
	QVector< double >                                    mDropoutRatios;				   //!< The ratio of dropout in the min-max range of weights of each layer's neuron.
	QVector< double >                                    mDropoutProbabilities;			   //!< The probability of dropout per layer.
	QVector< QVector< std::shared_ptr< DEBINeuron > > >  mDEBINeurons;					   //!< The DEBI neurons per layer.
	bool                                                 mIsRandomInitialize;			   //!< Stores if the network shall be randomly initialized upon instantiation.
	QVector< QPair< double, double > >                   mZRanges;						   //!< The min-max z-ranges of each layer.
	QStringList                                          mPredictionLabels;				   //!< The labels the network can predict.
	std::mt19937*                                        mGenerator;                       //!< Random generator to generate probabilistic values for random calculations.
	bool                                                 mIsRealigned;					   //!< Stores if the network is spatially aligned.
	bool                                                 mIsLayerDeltasOptimize;		   //!< Stores if the layer deltas shall be trainable parameters.
	bool                                                 mIsDropoutRatiosOptimize;		   //!< Stores if the dropout ratios shall be trainable parameters.
	bool                                                 mIsDropoutProbabilitiesOptimize;  //!< Stores if the dropout probabilities shall be trainable parameters.
	bool                                                 mIsWeightsStandardize;			   //!< Stores if input weights of a neuron shall be z-score standardized.
	bool                                                 mIsGroupNormalize;				   //!< Stores if group normalization shall take place.
	bool                                                 mIsShiftScaleOptimize;			   //!< Stores if shift-scale parameters per neuron group shall be optimized.
	double                                               mGlobalMinWeight;				   //!< The smallest weight in the network.
	double                                               mGlobalMaxWeight;				   //!< The largest weight in the network.
};

//-----------------------------------------------------------------------------

}


QDataStream& operator<<( QDataStream& aOut, const muw::DEBINN::NeuronGroup& aNeuronGroup );
QDataStream& operator>>( QDataStream& aIn, muw::DEBINN::NeuronGroup& aNeuronGroup );

//-----------------------------------------------------------------------------
