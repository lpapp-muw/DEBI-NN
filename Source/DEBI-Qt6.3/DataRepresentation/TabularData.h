/*!
* \file
* This file is part of Datarepresentation module.
*
* \remarks
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#pragma once

#include <DataRepresentation/Export.h>
#include <DataRepresentation/Types.h>
#include <QString>
#include <QVariant>
#include <QHash>
#include <QList>
#include <QDataStream>
#include <QVector>

// Forward declarations
namespace muw
{
	class TabularData;
}

QDataStream& operator<<( QDataStream &aOut, const muw::TabularData& aTabularData );
QDataStream& operator>>( QDataStream &aIn, muw::TabularData& aTabularData );

namespace muw
{

//-----------------------------------------------------------------------------

typedef QMap< QString , QVariant >      TabularDataHeader;
typedef QHash< QString, QVariantList >  TabularDataTable;

//-----------------------------------------------------------------------------

/*!
* \brief The TabularData class is a general container to hold a key and its respective values in tabular form.
*
* \details 
*/
class DataRepresentation_API TabularData
{

public:
	/*!
	* \brief Default constructor.
	*/
	TabularData();

	/*!
	* \brief Constructor.
	* \param [in] aName Name of the tabular data.
	*/
	TabularData( const QString& aName );

	/*!
	* \brief Copy constructor.
	* \param [in] aOther Object to copy.
	*/
	TabularData( const TabularData& aOther );

	/*!
	* \brief Move constructor.
	* \param [in] aOther Object to move.
	*/
	TabularData( TabularData&& aOther );

	/*!
	* \brief Destructor.
	*/
	~TabularData();

	/*!
	* \brief Returns with the hash representing the table itself.
	* \return The QHash containing the table.
	*/
	TabularDataTable& table() { return mTable; }

	/*!
	* \brief Returns with the hash representing the table itself.
	* \return The QHash containing the table.
	*/
	const TabularDataTable& table() const { return mTable; }

	/*!
	* \brief Returns with the map representing the table header.
	* \return The QMap containing the header.
	*/
	TabularDataHeader& header() { return mHeader; }

	/*!
	* \brief Returns with the map representing the table header.
	* \return The QMap containing the header.
	*/
	const TabularDataHeader& header() const { return mHeader; }

	/*!
	* \brief Sets the header names based on the input.
	* \param [in] aHeaderNames the string list containing the header names to be set.
	*/
	void setHeader( QStringList aHeaderNames );

	/*!
	* \brief Returns with the header names of the table.
	* \return The QStringList containing the header.
	*/
	const QStringList headerNames();

	/*!
	* \brief Returns with the row of the respected key.
	* \param [in] aKey The key of the requested value.
	* \return The value list of the key.
	*/
	const QVariantList value( const QString& aKey ) const { return mTable.value( aKey ); }

	/*!
	* \brief Returns with the row of the respected key.
	* \param [in] aKey The key of the requested value.
	* \return The value list of the key.
	*/
	QVariantList& value( const QString& aKey ) { return mTable[ aKey ]; }

	/*!
	* \brief Operator [] to return with the row of the respected key.
	* \param [in] aKey The key of the requested value.
	* \return The value list of the key.
	*/
	QVariantList& operator[]( const QString& aKey ) { return mTable[ aKey ]; }

	/*!
	* \brief Operator = to copy the contents of the input into self.
	* \param [in] aRight The tabular data to copy.
	* \return The new tabular data copied from the input.
	*/
	muw::TabularData& operator=(const muw::TabularData& aRight)
	{
		mTable  = aRight.mTable;
		mHeader = aRight.mHeader;
		mName   = aRight.mName;
		return *this;
	}

	/*!
	* \brief Returns with the value of the respected key and column index.
	* \param [in] aKey The key of the requested value.
	* \param [in] aColumnIndex The column index of the requested value.
	* \return The value of the key at the given column index.
	*/
	const QVariant valueAt( QString aKey, int aColumnIndex ) const { return mTable.value( aKey ).at( aColumnIndex ); }

	/*!
	* \brief Returns with the value of the respected key and column index.
	* \param [in] aKey The key of the requested value.
	* \param [in] aColumnIndex The column index of the requested value.
	* \return The value of the key at the given column index.
	*/
	QVariant& valueAt( QString aKey, int aColumnIndex ) { return mTable[ aKey ][ aColumnIndex ];
}

	/*!
	* \brief Inserts a new value identified with the key.
	* \param [in] aKey The key of the value to insert.
	* \param [in] aValue the value of the key to insert.
	*/
	void insert( const QString& aKey, const QVariantList& aValue ) { mTable.insert( aKey, aValue ); }

	/*!
	* \brief Removes all values associated with the input key.
	* \param [in] aKey The key of the value to remove.
	* \return The number of values removed.
	*/
	int remove( const QString& aKey ) { return mTable.remove( aKey ); }

	/*!
	* \brief Returns with the unique keys located in the table.
	* \return The list of the unique keys.
	*/
	QList< QString > keys() const { return mTable.keys(); }

	/*!
	* \brief Returns with the column as of the input column index.
	* \param [in] aColumnIndex The index of the column to return.
	* \return The list of values in the given column.
	*/
	QVariantList column( unsigned int aColumnIndex );

	/*!
	* \brief Returns with the column as of the input column name.
	* \param [in] aColumnName The name of the column to return.
	* \return The list of values in the given column.
	*/
	QVariantList column( QString aColumnName );

	/*!
	* \brief Returns with the column name as of the input column index.
	* \param [in] aColumnIndex The index of the column.
	* \return The name of the column as of in the given column index.
	*/
	QString columnName( int aColumnIndex ) const { return mHeader.value( QString::number( aColumnIndex ) ).toStringList().at( 0 ); }

	/*!
	* \brief Clears the table.
	*/
	void clear() { mTable.clear(); }

	/*!
	* \brief Returns with the row count of the table.
	* \return The number of rows in the table.
	*/
	unsigned int rowCount() const { return mTable.count(); }

	/*!
	* \brief Returns with the column count of the table.
	* \return The number of columns in the table.
	*/
	unsigned int columnCount() const { return mHeader.size(); }

	/*!
	* \brief Returns with the name of the table.
	* \return The name of the table.
	*/
	const QString& name() const { return mName; }

	/*!
	* \brief Returns with the name of the table.
	* \return The name of the table.
	*/
	QString& name() { return mName; }


	/*!
	* \brief Calculates the mean of the given column.
	* \param [in] aColumnIndex The column index for the mean calculation.
	* \return The mean of the given column.
	*/
	double mean( unsigned int aColumnIndex );

	/*!
	* \brief Calculates the standard deviation of the given column.
	* \param [in] aColumnIndex The column index for the standard deviation calculation.
	* \return The standard deviation of the given column.
	*/
	double deviation( unsigned int aColumnIndex );

	/*!
	* \brief Calculates the minimum of the given column.
	* \param [in] aColumnIndex The column index for the minimum calculation.
	* \return The minimum of the given column.
	*/
	double min( unsigned int aColumnIndex );

	/*!
	* \brief Calculates the maximum of the given column.
	* \param [in] aColumnIndex The column index for the maximum calculation.
	* \return The maximum of the given column.
	*/
	double max( unsigned int aColumnIndex );

	/*!
	* \brief Returns with the minimums of each column in the table.
	* \return The QVector containing the minimums of each column in the table.
	*/
	QVector< double > mins();

	/*!
	* \brief Returns with the maximums of each column in the table.
	* \return The QVector containing the maximums of each column in the table.
	*/
	QVector< double > maxs();

	/*!
	* \brief Returns with the means of each column in the table.
	* \return The QVariantList containing the means of each column in the table.
	*/
	QVariantList means();

	/*!
	* \brief Returns with the deviations of each column in the table.
	* \return The QVariantList containing the deviations of each column in the table.
	*/
	QVariantList deviations();

	/*!
	* \brief Merges with the input tabular data by adding its records. Merge is only performed if the features of the input tabular data and the own data are identical.
	* \param [in] aTabularData The input table to merge with.
	*/
	void mergeRecords( muw::TabularData& aTabularData );

	/*!
	* \brief Merges with the input tabular datas by adding their features. Merge is only performed if the samples of the input tabular data and the own data have the same keys.
	* \param [in] aTabularData The input table to merge with.
	*/
	static muw::TabularData mergeFeatures( QList< muw::TabularData > aTabularDatas );

	/*!
	* \Brief operator << for QDataStream compatibility.
	* \param [in] aOut The output to stream the contents of the table to.
	* \param [in] aTabularData the source of streaming.
	* \return The updated QDataStream.
	*/
	friend QDataStream& ::operator<<( QDataStream &aOut, const TabularData& aTabularData );

	/*!
	* \Brief operator >> for QDataStream compatibility.
	* \param [in] aIn The input to stream to read out the contents to the table.
	* \param [in] aTabularData tthe destination of streaming.
	* \return The updated QDataStream.
	*/
	friend QDataStream& ::operator>>( QDataStream &aIn, TabularData& aTabularData );


protected:
	muw::TabularDataTable   mTable;   //!< The table containing the key and a respective variant list.
	muw::TabularDataHeader  mHeader;  //!< The table header containing the names and types of the columns. Key column is not taken into account.
	QString                 mName;    //!< The name of the tabular data.

};

template< typename T >
QSet< T > toSet( const QList< T >& aList )
{
	return QSet< T >( aList.begin(), aList.end() );
}

//-----------------------------------------------------------------------------

}
