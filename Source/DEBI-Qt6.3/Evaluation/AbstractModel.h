/*!
* \file
* This file is part of Evaluation module.
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
#include <QSettings>
#include <QVariant>
#include <QVector>
#include <QString>
#include <QDataStream>

namespace muw
{

//-----------------------------------------------------------------------------

/*!
* \brief The AbstractModel class is a base class for machine learning model implementations in order to facilitate a generic mechanism for interacting with models, regardless of their actual implementaiton.
*
* \details Note that while settings in constructors suppose to initialize the model and/or to define their architectures, providing actual parameters for a model is supposed to come during a training process (from an Optimizer) and intends to define parameters of an already existing model.
* Example: In a neural network model, the number of hidden layers and their neuron counts or e.g., activation types suppose to come from settings and shall be provided in the constructor. In contrast, weights shall get set by the set() function during training.
* Particular implementations may rely on either constructor settings or set() calls or both. Building e.g., random forests may not require calling set(), if an Optimizer (tree builder) builds the random forest model entirely.
*/

class Evaluation_API AbstractModel
{

public:

	/*!
	* \brief Constructor.
	* \param [in] aSettings the settings container to load parameters from for initializing the model.
	*/
	AbstractModel( QSettings* aSettings );

	/*!
	* \brief Sets the input parameters of the given model. This function is utilized by the Optimizer during training processes to modify an existing model.
	* \param [in] aParameters the list of parameters for the model.
	*/
	virtual void set( const QVector< double >& aParameters ) = 0;

	/*!
	* \brief Evaluates the input feature vector and returns with a prediction.
	* \param [in] aFeatureVector the feature vector containing features to evaluate for predicting.
	* \return the prediction of the model which can have a user-defined format. In case the model is a classifier, it shall return with a label-probability map, describing all possible label outcomes so that classifier analytics (e.g., confusion matrix can operate with them).
	*/
	virtual QVariant evaluate( const QVector< double >& aFeatureVector ) = 0;

	/*!
	* \return the number of input parameters the model requires to configure itself.
	*/
	virtual int inputCount() = 0;

	/*!
	* \brief Destructor.
	*/
	virtual ~AbstractModel();

	/*!
	* \return the feature names the given model can accept as inputs.
	*/
	const QList< QString >& featureNames() const { return mFeatureNames; }

	/*!
	* \return the reference of feature names the given model can accept as inputs.
	*/
	QList< QString >&       featureNames() { return mFeatureNames; }

private:

	/*!
	* \brief Default contructor not allowed.
	*/
	AbstractModel();

protected:

	QSettings*        mSettings;      //!< The settings container to initialize the model.
	QList< QString >  mFeatureNames;  //!< The feature names the given model is able to predict from and/or learn from.
};

//-----------------------------------------------------------------------------

}

