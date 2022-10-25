/*!
* \file
* This file is part of the TstApplication module.
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
#include <Evaluation/DataPackage.h>
#include <DataRepresentation/TabularData.h>
#include <QString>
#include <QMap>
#include <QVariant>
#include <QVector>


namespace muw
{

//-----------------------------------------------------------------------------

/*!
* \brief The DEBINNCrossValidator class is responsible to build and cross-validate DEBINN models based on the input datasets and settings provided.
*
* \details In case multiple settings are provided, the DEBINNCrossValidator class trains and cross-validates DEBINN models by all of the settings sequentially. This mechanism may be relied on in case hyperparameter optimization is performed.
* The input dataset has to be organized in a folder structure containing "Fold-X" subfolders (where X is a postive integer and fold subfolders are to be numbered sequentially). In each fold folder there have to be 4 files: TDS, VDS, TLD and VLD describing the training feature database, the validation feature database, the training label database and the validation label tabase respectively.
* Note that this class does not perform data preprocessing on the loaded datasets, but process them as-is. This means that data prprocessing steps (e.g., redundancy reduction, feature ranking and selection, z-score standardization) are to be performed prior to utilizing this class.
*
*/

class DEBINNCrossValidator
{

public:

	/*!
	\brief Constructor.
	\param [in] aDataFoldsFolderPath the folder in which the folds (Fold-1, Fold-2, ...) are containing datasets for training and validating DEBINN models.
	\param [in] aDEBINNFactorySettingFolderPath the folder in which the setting file is or files are to build and cross-validate DEBINN models.
	*/
	DEBINNCrossValidator( QString aExecutionFolderPath );
	
	/*!
	\brief Destructor.
	*/
	~DEBINNCrossValidator();

	/*!
	\brief Executes the training and cross-validation process.
	*/
	void execute();

private:

	/*!
	\brief Default constructor not allowed.
	*/
	DEBINNCrossValidator();

	/*!
	\brief Executes the training and cross-validation of DEBINN models of the given cross-validation fold.
	\param [in] aFoldFolderPath the folder in which the datasets for training and validating DEBINN models of the given fold are.
	*/
	void executeFold( QString aFoldFolderPath );


	//void resetInternalDataSets();

private:

	QString                  mExecutionFolderPath;	   //!< The folder in which the folds for training and cross-validaiton are.
	QVector< QSettings* >    mSettingsVariants;		   //!< The settings loaded up to train DEBINN models.
	QMap< QString, double >  mCrossValidationResults;  //!< The predictive performance results of cross-validation.
	int                      mFoldCount;			   //!< The number of folds identified for cross-validaiton.
	muw::TabularData         mLogTable;				   //!< The log container to store training and validation errors across training and cross-validating DEBINN models.
	muw::TabularData         mResultTable;			   //!< The tabular data to save out the log container content.
	

};

//-----------------------------------------------------------------------------

}
