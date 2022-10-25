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
#include <Evaluation/DEBINN.h>
#include <Evaluation/DataPackage.h>
#include <QString>
#include <QSettings>


namespace muw
{

//-----------------------------------------------------------------------------

/*!
* \brief The DEBINNFactory class is a convenience implementation to load up, initialize, train, and log training processes of DEBI neural networks.
*
* \details
*/

class Evaluation_API DEBINNFactory
{

public:

	/*!
	\brief Default constructor.
	*/
	DEBINNFactory();

	/*!
	\brief Constructor.
	\param [in] aSettings the settings container which has the setting parameters to initialize the DEBI neural network.
	*/
	DEBINNFactory( QSettings* aSettings );

	/*!
	\brief Constructor.
	\param [in] aSettingsFolderPath the path of the settings file holding the parameters to initialize the DEBI neural network.
	*/
	DEBINNFactory( QString aSettingsFolderPath );

	/*!
	\brief Destructor.
	*/
	~DEBINNFactory();

	/*!
	\brief Generates a trained DEBI neural network based on its input data package which acts as training set. If a second data package is also included, it will be subject of an internal evaluation in order to log training-validation errors across training. Note that the validation set is not taken part in any decision making of the training process.
	\param [in] aSettings the settings that contains parameters to initialize a DEBI neural network
	*/
	std::shared_ptr< muw::DEBINN > generate( muw::DataPackage* aDataPackage, muw::DataPackage* aValidationDataPackage = nullptr );

	/*!
	\brief Generates a trained DEBI neural network by loading it up from a saved file.
	\param [in] aDEBIModelFolderPath the path to the binary file containing a trained DEBI neural network.
	*/
	std::shared_ptr< muw::DEBINN > generate( QString aDEBIModelFolderPath );

	/*!
	\return The log container which was created during a DEBI neural network training process.
	*/
	QVariantMap log() { return mLog; }

private:

	/*!
	\brief Initializes a DEBI neural network based on the input settings.
	\param [in] aSettings the settings containing parameters to initialize a DEBI neural network.
	*/
	void initialize( QSettings* aSettings );


private:

	QVariantMap        mSettings;     //!< The settings container to initialize a DEBI neural network.
	muw::DataPackage*  mDataPackage;  //!< The datapackage to train a DEBI neural network.
	QVariantMap        mLog;          //!< The log container created during training a DEBI neural network.

};

//-----------------------------------------------------------------------------

}
