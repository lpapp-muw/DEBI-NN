/*!
* \file
* This file is part of FileIo module.
* Member function definitions for TabularDataFileIo class.
*
* \remarks
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#include <FileIo/TabularDataFileIo.h>
#include <QFile>
#include <QTextStream>
#include <QJsonObject>
#include <QJsonDocument>
#include <QDebug>

namespace muw
{

//-----------------------------------------------------------------------------

TabularDataFileIo::TabularDataFileIo()
:
	mWorkingDirectory( "" ),
	mCommaSeparator( ';' )
{
}

//-----------------------------------------------------------------------------

TabularDataFileIo::TabularDataFileIo( QString aWorkingDirectory )
:
	mWorkingDirectory( aWorkingDirectory ),
	mCommaSeparator( ';' )
{
}

//-----------------------------------------------------------------------------

TabularDataFileIo::~TabularDataFileIo()
{
}

//-----------------------------------------------------------------------------

void TabularDataFileIo::load( QString aFileName, muw::TabularData& aTabularData )
{

	QString fullPath;
	if ( mWorkingDirectory.count() == 0 )
	{
		fullPath = aFileName;
	}
	else
	{
		fullPath = mWorkingDirectory + "/" + aFileName;
	}
	
	//Load CSV
	if ( !fullPath.contains( ".csv" ) )
	{
		fullPath = fullPath + ".csv";
	}
	QFile fileInCsv( fullPath );
	if ( fileInCsv.open( QIODevice::ReadOnly | QIODevice::Text) )
	{

		QTextStream in( &fileInCsv );

		QString key;
		QVariantList row;

		// Read the header
		QString line = in.readLine();  //Read the header data which is the first line.
		QList< QString > separatedLine = line.split( mCommaSeparator );  //Separate column values.

		if ( QString( separatedLine.at( separatedLine.size() - 1 ) ).compare("") == 0 )
		{
			separatedLine.removeLast();
		}
		
		// Remove possible empty strings from end.
		muw::TabularDataHeader header;

		for ( int headerIndex = 1; headerIndex < separatedLine.size(); ++headerIndex )
		{
			QString value = separatedLine.at( headerIndex );
			QString type = "Float";
			QVariantList headerValue = { value, type };
			header.insert( QString::number( headerIndex - 1), headerValue );
		}

		aTabularData.header() = header;  // Save the header to the tabular data.

		// Read up the file.
		while ( !in.atEnd() )
		{
			
			QString line = in.readLine();  // Read one line, remove whitespaces.
			QList< QString > separatedLine = line.split( mCommaSeparator );  // Separate column values.
			key = separatedLine.at( 0 );  // Read out the key.

			// Fill up the row with values of the given key.
			row.clear();
			for ( int columnIndex = 1; columnIndex < separatedLine.size(); ++columnIndex )
			{
				row.push_back( separatedLine.at( columnIndex ) );
			}

			// Place the key-value pairs into the tabular data.
			aTabularData.insert( key, row );
		}

		fileInCsv.close();

		//qDebug() << "Loading finished." << endl;
	}
	else qDebug() << "Failed to open: " << fullPath << Qt::endl;
}

//-----------------------------------------------------------------------------

void TabularDataFileIo::save( QString aFileName, muw::TabularData& aTabularData )
{
	QString fullPath;
	if ( mWorkingDirectory.count() == 0 )
	{
		fullPath = aFileName;
	}
	else
	{
		fullPath = mWorkingDirectory + "/" + aFileName;
	}

	// Save CSV
	if ( !fullPath.contains( ".csv" ) )
	{
		fullPath = fullPath + ".csv";
	}
	QFile fileOutCsv( fullPath );
	if ( fileOutCsv.open( QFile::WriteOnly | QFile::Text ) )
	{
		QTextStream stream( &fileOutCsv );

		// Save header.
		QMap< QString, QVariant > header = aTabularData.header();

		stream << "Key";

		for ( int headerIndex = 0; headerIndex < header.size(); ++headerIndex )
		{
			QStringList headerColumnDescriptor = header.value( QString::number( headerIndex ) ).toStringList();
			stream << mCommaSeparator << headerColumnDescriptor.at( 0 );
		}

		stream << Qt::endl;
		fileOutCsv.flush();
		

		// Save the tabular data entries.
		auto keys = aTabularData.keys();
		std::sort( keys.begin(), keys.end() );

		for ( auto key : keys )
		{
			stream << key;
			for ( int columnIndex = 0; columnIndex < aTabularData.columnCount(); ++columnIndex )
			{
				auto cell = aTabularData.valueAt( key, columnIndex ).toString();
				if ( cell == "nan" )
				{
					cell = "NA";
				}
				stream << mCommaSeparator << cell;
			}
			
			stream << Qt::endl;
			fileOutCsv.flush();
		}

		fileOutCsv.close();
	}
	else
	{
		qDebug() << "Cannot open for read: " << fullPath;
	}
}

//-----------------------------------------------------------------------------

}
