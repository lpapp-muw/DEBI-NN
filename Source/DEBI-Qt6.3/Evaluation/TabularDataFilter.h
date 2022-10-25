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
#include <Evaluation/DataPackage.h>
#include <DataRepresentation/TabularData.h>
#include <QDebug>
#include <random>

namespace muw
{

//-----------------------------------------------------------------------------

/*!
* \brief The TabularDataFilter class is a convenience implementation of several filtering steps to manipulate or extract information from TabularData objects.
*
* \details
*/

class Evaluation_API TabularDataFilter
{

public:

	/*!
	* \brief Default constructor.
	*/
	TabularDataFilter();

	/*!
	* \brief Destructor.
	*/
	virtual ~TabularDataFilter() {}

	/*!
	* \brief Erases records from the dabular data if there are features in them with missing values.
	* \param [in, out] aFeatureDatabase the tabular data to process.
	*/
	void eraseIncompleteRecords( muw::TabularData& aFeatureDatabase );

	/*!
	* \brief Returns the table of features that have the given label subgroup.
	* \param [in] aFeatureDatabase the tabular dataset containing the feature vectors.
	* \param [in] aLabelDatabase the tabular dataset containing the label vectors.
	* \param [in] aLabelIndex the column index in the label database to take the current label subgroups from. Shall be 0 in case only one label column is in the label database.
	* \param [in] aReferenceLabel the reference value of the given label subgroup within the active label column to filter.
	* \return the table containing only features having aReferenceLabel values in the corresponding label database.
	*/
	muw::TabularData subTableByLabelGroup( const muw::TabularData& aFeatureDatabase, const muw::TabularData& aLabelDatabase, const int aLabelIndex, const QString aReferenceLabel );

	/*!
	* \brief Returns the keys of features that have the given label subgroup.
	* \param [in] aLabelDatabase the tabular dataset containing the label vectors.
	* \param [in] aLabelIndex the column index in the label database to take the current label subgroups from. Shall be 0 in case only one label column is in the label database.
	* \param [in] aReferenceLabel the reference value of the given label subgroup within the active label column to filter.
	* \return the keys of features having aReferenceLabel values in the corresponding label database.
	*/
	QStringList keysByLabelGroup( const muw::TabularData& aLabelDatabase, const int aLabelIndex, const QString aReferenceLabel );

	/*!
	* \brief Returns the list of label groups of the given label.
	* \param [in] aLabelDatabase the tabular dataset containing the label vectors.
	* \param [in] aLabelIndex the column index in the label database to take the current label subgroups from. Shall be 0 in case only one label column is in the label database.
	* \param [in] aIsNAIncluded if true, missing values will have their own label subgroup. If false, missing values will be skipped.
	* \return the label subgroup names of the given label.
	*/
	QStringList labelGroups( const muw::TabularData& aLabelDatabase, const int aLabelIndex, bool aIsNAIncluded = false );

	/*!
	* \brief Returns the table containing records with the input keys only.
	* \param [in] aTabularData the tabular dataset to filter from.
	* \param [in] aReferenceKeys the list of keys to search for to generate the subtable.
	* \return the table containing records of the input keys only.
	*/
	muw::TabularData subTableByKeys( const muw::TabularData& aTabularData, QStringList aReferenceKeys );

	/*!
	* \brief Identifies the common keys of the feature and label databases as of the current label.
	* \param [in] aFeatureDatabase the tabular dataset containing the feature vectors.
	* \param [in] aLabelDatabase the tabular dataset containing the label vectors.
	* \param [in] aLabelIndex the column index in the label database to take the current label subgroups from. Shall be 0 in case only one label column is in the label database.
	* * \param [in] aIsNAIncluded if true, keys having missing values as of aLabelIndex will be part of the common key identification process. If false, keys corresponding to missing label values will be skipped.
	* \return The common keys present in both feature and label databases.
	*/
	QStringList commonKeys( const muw::TabularData& aFeatureDatabase, const muw::TabularData& aLabelDatabase, const int aLabelIndex, bool aIsNAIncluded = false );


private:

};

//-----------------------------------------------------------------------------

}
