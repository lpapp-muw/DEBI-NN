/* !
*\file
* This file is part of Evaluation module.
* Member class definitions of TabularDataFilter.
*
* \remarks
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#include <Evaluation/TabularDataFilter.h>
#include <QSet>
#include <omp.h>
#include <algorithm>
#include <QDebug>

using muw::toSet;

namespace muw
{

//-----------------------------------------------------------------------------

TabularDataFilter::TabularDataFilter()
{
}

//-----------------------------------------------------------------------------

void TabularDataFilter::eraseIncompleteRecords( muw::TabularData& aFeatureDatabase )
{
	auto keys = aFeatureDatabase.keys();
	QStringList keysToDelete;

	for ( auto key : keys )
	{
		auto featureVector = aFeatureDatabase.value( key );
		if ( featureVector.contains( "NA" ) || featureVector.contains( "nan" ) )
		{
			keysToDelete.push_back( key );
		}
	}

	for ( auto key : keysToDelete )
	{
		aFeatureDatabase.remove( key );
	}
}

//-----------------------------------------------------------------------------

muw::TabularData TabularDataFilter::subTableByLabelGroup( const muw::TabularData& aFeatureDatabase, const muw::TabularData& aLabelDatabase, const int aLabelIndex, const QString aReferenceLabel )
{
	QStringList keys = commonKeys( aFeatureDatabase, aLabelDatabase, aLabelIndex );  // Determine common keys.

	// Construct the subset tabular data.
	muw::TabularData subset;
	subset.header() = aFeatureDatabase.header();

	for ( int keyIndex = 0; keyIndex < keys.size(); ++keyIndex )
	{
		QString actualKey = keys.at( keyIndex );
		QString actualLabel = aLabelDatabase.valueAt( actualKey, aLabelIndex ).toString(); // Read out the label from the column of aLabelDatabase deterlined by the aLabelIndex.

		if ( actualLabel == aReferenceLabel )  // We found a label matching the reference label.
		{
			subset.insert( actualKey, aFeatureDatabase.value( actualKey ) );  // Save the given row.
		}
	}

	return subset;
}

//-----------------------------------------------------------------------------

QStringList TabularDataFilter::keysByLabelGroup( const muw::TabularData& aLabelDatabase, const int aLabelIndex, const QString aReferenceLabel )
{
	QStringList keys = aLabelDatabase.keys();
	QStringList keysByLabel;

	for ( int keyIndex = 0; keyIndex < keys.size(); ++keyIndex )
	{
		QString actualKey = keys.at( keyIndex );
		QString actualLabel = aLabelDatabase.valueAt( actualKey, aLabelIndex ).toString(); // Read out the label from the column of aLabelDatabase deterlined by the aLabelIndex.

		if ( actualLabel == aReferenceLabel )  // We found a label matching the reference label.
		{
			keysByLabel.push_back( actualKey );
		}
	}

	return keysByLabel;
}

//-----------------------------------------------------------------------------

QStringList TabularDataFilter::labelGroups( const muw::TabularData& aLabelDatabase, const int aLabelIndex, bool aIsNAIncluded )
{
	QStringList labelsOfDatabase;
	QStringList keys = aLabelDatabase.keys();

	for ( int rowIndex = 0; rowIndex < keys.size(); ++rowIndex )
	{
		QString actualLabel = aLabelDatabase.valueAt( keys.at( rowIndex ), aLabelIndex ).toString();

		if ( actualLabel == "NA" && aIsNAIncluded == false ) continue;

		if ( !labelsOfDatabase.contains( actualLabel ) )
		{
			labelsOfDatabase.push_back( actualLabel );
		}
	}

	std::sort( labelsOfDatabase.begin(), labelsOfDatabase.end() );

	return labelsOfDatabase;
}

//-----------------------------------------------------------------------------

muw::TabularData TabularDataFilter::subTableByKeys( const muw::TabularData& aTabularData, QStringList aReferenceKeys )
{
	muw::TabularData subsetByKeyTabularData;
	subsetByKeyTabularData.header() = aTabularData.header();

	QStringList tabularDataKeys = aTabularData.keys();
	QStringList commonKeys = toSet( tabularDataKeys ).intersect( toSet( aReferenceKeys ) ).values();  // Determine common keys.

	for ( int keyIndex = 0; keyIndex < commonKeys.size(); ++keyIndex )
	{
		QString actualKey = commonKeys.at( keyIndex );
		QVariantList row = aTabularData.value( actualKey );
		subsetByKeyTabularData.insert( actualKey, row );
	}

	return subsetByKeyTabularData;
}

//-----------------------------------------------------------------------------

QStringList TabularDataFilter::commonKeys( const muw::TabularData& aFeatureDatabase, const muw::TabularData& aLabelDatabase, const int aLabelIndex, bool aIsNAIncluded )
{
	QSet< QString > featureKeys = toSet( aFeatureDatabase.keys() );
	QSet< QString > labelKeys = toSet( aLabelDatabase.keys() );
	QStringList NAKeys = keysByLabelGroup( aLabelDatabase, aLabelIndex, "NA" );

	if ( !NAKeys.isEmpty() && !aIsNAIncluded )
	{
		labelKeys.subtract( toSet( NAKeys ) );
	}

	return featureKeys.intersect( labelKeys ).values();
}

//-----------------------------------------------------------------------------

}
