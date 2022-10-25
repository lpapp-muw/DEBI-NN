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
#include <DataRepresentation/TabularData.h>
#include <QSettings>
#include <QString>
#include <QDataStream>

namespace muw
{

//-----------------------------------------------------------------------------

/*!
* \brief The DataPackage class contains a feature and a corresponding label tabular database.
* 
* \details The DataPackage class requires to store feature and label databases separately. Relying on this mechanism minimizes the risk of label leakage into the training process as a feature.
* Since DataPackage always requires a feature and label database, it requires the users to prepare such databases separately, where the given feature database shall not contain labels to predict in the given training context.
* Due to the existence of feature and label databases, records in-between them are matched by their Key column contents.
* The DataPackage class has various convenience functions to filter its records, columns or to provide the list of label outcomes and their specific samples. Accessing both feature vectors and their respective labels is provided by unique Key identifiers being present in both databases.
*
*/

class Evaluation_API DataPackage
{

public:

	/*!
	* \brief Constructor.
	* \param [in] aSettings the setting container describing the parameters of creating a data package.
	*/
	DataPackage( QSettings* aSettings );

	/*!
	* \brief Constructor.
	* \param [in] aFDB the input feature database.
	* \param [in] aLDB the input label database.
	* \param [in] aLableName the name of the label column to be active in the data package.
	* \param [in] aIncludedKeys the list of keys to be included from the feature and label databases. In case it is empty, it is not considered for filtering.
	*/
	DataPackage( muw::TabularData& aFDB, muw::TabularData& aLDB, QString aLabelName, QStringList aIncludedKeys = {} );

	/*!
	* \Brief Destructor.
	*/
	virtual ~DataPackage();

	/*!
	* \return the feature database of the data package.
	*/
	muw::TabularData& featureDB() { return mFDB; }

	/*!
	* \return the label database of the data package.
	*/
	muw::TabularData& labelDB() { return mLDB; }

	/*!
	* \return the number of features in the feature database.
	*/
	const int featureCount() const { return mFeatureCount; }

	/*!
	* \return the name of the active label column in the label database.
	*/
	const QString& labelName() const { return mLabelName; }

	/*!
	* \return the index of the active label column in the label database.
	*/
	const int labelIndex() const { return mLabelIndex; }

	/*!
	* \return the list of label outcomes of the active label in the label database.
	*/
	const QStringList labelOutcomes() { return mLabelOutcomes; }

	/*!
	* \return the list of keys of the feature database.
	*/
	const QStringList sampleKeys() { return mSampleKeys; }

	/*!
	* \brief returns ture if the data package is in a valid state. A data package may be in an invalid state if e.g., the feature and label database provided to it have no common keys.
	* \return true of the data package is in a valid state. False otherwise.
	*/
	const bool isValid() { return mIsValidDataset; }

private:

	/*!
	* \brief initializes the data package as of the input parameters.
	* \param [in] aFDB the input feature database.
	* \param [in] aLDB the input label database.
	* \param [in] aLableName the name of the label column to be active in the data package.
	*/
	void initialize( muw::TabularData& aFDB, muw::TabularData& aLDB, QString aLabelName );

protected:

	QSettings*        mSettings;        //!< Settings container to instantiate a data package.
	muw::TabularData  mFDB;             //!< The feature database.
	muw::TabularData  mLDB;             //!< The label database.
	int               mFeatureCount;    //!< The number of features in the feature database.
	QString           mLabelName;       //!< The nam of the active column in the label database.
	QStringList       mIncludedKeys;;   //!< The list of keys to select from the input feature dataset to include in the data package.
	int               mLabelIndex;      //!< The column index of the active column in the label database.
	QList< QString >  mLabelOutcomes;   //!< The list of label subgroups in the label database as of the active label column.
	QList< QString >  mSampleKeys;      //!< The list of keys in the feature database.
	bool              mIsValidDataset;  //!< Stores if the data package is valid.
};

//-----------------------------------------------------------------------------

}
