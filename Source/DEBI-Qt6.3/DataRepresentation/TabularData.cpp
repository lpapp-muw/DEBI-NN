/*!
* \file
* This file is part of DataRepresentation module.
* Member function definitions for TabularData class.
*
* \remarks
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#include <DataRepresentation/TabularData.h>
#include <cfloat>
#include <cmath>
#include <QDebug>

namespace muw
{

//-----------------------------------------------------------------------------

TabularData::TabularData()
	:
	mTable(),
	mHeader(),
	mName()
{
}

//-----------------------------------------------------------------------------

TabularData::TabularData( const QString& aName )
:
	mTable(),
	mHeader(),
	mName( aName )
{
}

//-----------------------------------------------------------------------------

TabularData::TabularData( const TabularData& aOther )
: 
	mTable( aOther.mTable ),
	mHeader( aOther.mHeader ),
	mName( aOther.mName ) 
{
}

//-----------------------------------------------------------------------------

TabularData::TabularData( TabularData&& aOther )
: 
	mTable( std::move( aOther.mTable ) ),
	mHeader( aOther.mHeader ),
	mName( aOther.mName )
{
}

//-----------------------------------------------------------------------------

TabularData::~TabularData()
{

	mTable.clear();
	mHeader.clear();
}

//-----------------------------------------------------------------------------

void TabularData::setHeader( QStringList aHeaderNames )
{
	muw::TabularDataHeader header;
	
	for ( int headerIndex = 0; headerIndex < aHeaderNames.size(); ++headerIndex )
	{
		QString value = aHeaderNames.at( headerIndex );
		QString type = "Float";
		QVariantList headerValue = { value, type };
		header.insert( QString::number( headerIndex ), headerValue );
	}

	mHeader = header;
}

//-----------------------------------------------------------------------------

QVariantList TabularData::column( unsigned int aColumnIndex )
{
	QVariantList column;

	QStringList keys = mTable.keys();

	for ( int keyIndex = 0; keyIndex < keys.count(); ++keyIndex )
	{
		column << mTable.value( keys.at( keyIndex ) ).at( aColumnIndex );
	}

	return column;
}

//-----------------------------------------------------------------------------

QVariantList TabularData::column( QString aColumnName )
{
	QStringList headerNames = this->headerNames();

	int headerIndex = headerNames.indexOf( aColumnName );

	return column( headerNames.indexOf( aColumnName ) );

}

//-----------------------------------------------------------------------------

const QStringList TabularData::headerNames()
{
	QStringList headerNames;

	for ( int headerIndex = 0; headerIndex < mHeader.size(); ++headerIndex )
	{
		headerNames << mHeader.value( QString::number( headerIndex ) ).toStringList().at( 0 );
	}

	return headerNames;
}

//-----------------------------------------------------------------------------

double TabularData::mean( unsigned int aColumnIndex )
{
	QVariantList col = column( aColumnIndex );

	double mean = 0.0;

	for ( int rowIndex = 0; rowIndex < col.size(); ++rowIndex )
	{
		mean += col.at( rowIndex ).toDouble();
	}

	return mean / col.size();
}

//-----------------------------------------------------------------------------

double TabularData::deviation( unsigned int aColumnIndex )
{
	QVariantList col = column( aColumnIndex );
	double deviation = 0.0;
	double meanOfColumn = mean( aColumnIndex );

	for ( int rowIndex = 0; rowIndex < col.size(); ++rowIndex )
	{
		deviation += ( col.at( rowIndex ).toDouble() - meanOfColumn ) * ( col.at( rowIndex ).toDouble() - meanOfColumn );
	}

	return sqrt( deviation / col.size() );
}

//-----------------------------------------------------------------------------

double TabularData::min( unsigned int aColumnIndex )
{
	QVariantList col = column( aColumnIndex );
	double min = DBL_MAX;

	for ( int rowIndex = 0; rowIndex < col.size(); ++rowIndex )
	{
		double actVal = col.at( rowIndex ).toDouble();
		if ( actVal < min ) min = actVal;
	}

	return min;
}

//-----------------------------------------------------------------------------

double TabularData::max( unsigned int aColumnIndex )
{
	QVariantList col = column( aColumnIndex );
	double max = -DBL_MAX;

	for ( int rowIndex = 0; rowIndex < col.size(); ++rowIndex )
	{
		double actVal = col.at( rowIndex ).toDouble();
		if ( actVal > max ) max = actVal;
	}

	return max;
}

//-----------------------------------------------------------------------------

QVector< double > TabularData::mins()
{
	QVector< double > mins;

	for ( int colIndex = 0; colIndex < columnCount(); ++colIndex )
	{
		mins.push_back( this->min( colIndex ) );
	}

	return mins;
}

//-----------------------------------------------------------------------------

QVector< double > TabularData::maxs()
{
	QVector< double > maxs;

	for ( int colIndex = 0; colIndex < columnCount(); ++colIndex )
	{
		maxs.push_back( this->max( colIndex ) );
	}

	return maxs;
}

//-----------------------------------------------------------------------------

QVariantList TabularData::means()
{
	QVariantList meanList;

	for ( unsigned int headerIndex = 0; headerIndex < mHeader.size(); ++headerIndex )
	{
		meanList.push_back( mean( headerIndex ) );
	}

	return meanList;
}

//-----------------------------------------------------------------------------

QVariantList TabularData::deviations()
{
	QVariantList deviationList;

	for ( unsigned int headerIndex = 0; headerIndex < mHeader.size(); ++headerIndex )
	{
		deviationList.push_back( deviation( headerIndex ) );
	}

	return deviationList;
}

//-----------------------------------------------------------------------------

void TabularData::mergeRecords( muw::TabularData& aTabularData )
{
	auto inputHeaderNames = aTabularData.headerNames();
	auto selfHeaderNames  = this->headerNames();

	QStringList commonFeatures = toSet( inputHeaderNames ).intersect( toSet( selfHeaderNames ) ).values();
	if ( commonFeatures.size() != inputHeaderNames.size() )
	{
		qDebug() << "TabularData merge - ERROR: Input and current features are not the same: ";
		qDebug() << "Input header names: " << inputHeaderNames;
		qDebug() << "Self header names: " << selfHeaderNames;
		return;
	}

	QStringList keys = aTabularData.keys();
	
	for ( int keyIndex = 0; keyIndex < keys.size(); ++keyIndex )
	{
		QString key = keys.at( keyIndex );
		QVariantList values = aTabularData.value( key );

		this->insert( key, values );
	}
}

//-----------------------------------------------------------------------------

muw::TabularData TabularData::mergeFeatures( QList< muw::TabularData > aTabularDatas )
{
	muw::TabularData mergedTabularData;

	// Read out the keys and the feature names.
	QList< QStringList > keysPerTabularData;
	QList< QStringList > featuresPerTabularData;
	for ( auto tabularData : aTabularDatas )
	{
		keysPerTabularData.push_back( tabularData.keys() );
		featuresPerTabularData.push_back( tabularData.headerNames() );
	}

	// Determine merged list of keys.
	QStringList mergedKeys;
	for ( auto keys : keysPerTabularData )
	{
		mergedKeys += keys;
		mergedKeys.removeDuplicates();
	}

	// Read merged list of feature names and set up the header of the merged table.
	QStringList mergedFeatureNames;
	for ( auto featureNames : featuresPerTabularData )
	{
		mergedFeatureNames += featureNames;
	}
	mergedTabularData.setHeader( mergedFeatureNames );

	// Create merged tabular data.
	for ( auto key : mergedKeys )
	{
		QVariantList mergedFeatureVector;
		for ( auto tabularData : aTabularDatas )
		{
			int featureCount = tabularData.columnCount();
			auto featureVector = tabularData.value( key );

			if ( featureVector.size() == 0 )  // This key is not in the given table
			{
				for ( int i = 0; i < featureCount; ++i )
				{
					featureVector.push_back( "NA" );
				}
			}

			mergedFeatureVector += featureVector;  // Merge the actual feature vector into the large feature vector.
		}

		mergedTabularData.insert( key, mergedFeatureVector );
	}


	return mergedTabularData;
}

//-----------------------------------------------------------------------------

}


QDataStream& operator<<( QDataStream &aOut, const muw::TabularData& aTabularData )
{
	aOut << aTabularData.mTable
			<< aTabularData.mHeader
			<< aTabularData.mName;

	return aOut;
}

QDataStream& operator>>( QDataStream &aIn, muw::TabularData& aTabularData )
{
	aIn >> aTabularData.mTable
		>> aTabularData.mHeader
		>> aTabularData.mName;

	return aIn;
}

