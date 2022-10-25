/*!
* \file
* This file is part of the Evaluation module.
* Member class definitions of DataPackage.
*
* \remarks
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#include <Evaluation/DataPackage.h>
#include <Evaluation/TabularDataFilter.h>
#include <FileIo/TabularDataFileIo.h>
#include <QFile>
#include <QDebug>

namespace muw
{

//-----------------------------------------------------------------------------

DataPackage::DataPackage( QSettings* aSettings )
:
	mSettings( aSettings ),
	mFDB(),
	mLDB(),
	mFeatureCount(),
	mLabelName(),
	mIncludedKeys(),
	mLabelIndex( 0 ),
	mLabelOutcomes(),
	mSampleKeys(),
	mIsValidDataset( false )
{
	// Read out the required data from the aSettings file.
	QString FDBPath      = mSettings->value( "DataPackage/FDBPath" ).toString();
	QString LDBPath      = mSettings->value( "DataPackage/LDBPath" ).toString(); 
	mLabelName           = mSettings->value( "DataPackage/LabelName" ).toString();
	QString includedPath = mSettings->value( "DataPackage/IncludedPath" ).toString();

	auto settingsPath = aSettings->fileName().split( "_MLAgents-Trained" ).at( 0 );

	FDBPath = settingsPath + "/_Folds/DataSource/" + FDBPath.split( "_Folds/DataSource/" ).at( 1 );
	LDBPath = settingsPath + "/_Folds/DataSource/" + LDBPath.split( "_Folds/DataSource/" ).at( 1 );

	muw::TabularDataFileIo fileIo;
	fileIo.load( FDBPath, mFDB );
	fileIo.load( LDBPath, mLDB );

	if ( !includedPath.isEmpty() )
	{
		QFile includedFile( includedPath );
		if ( includedFile.open( QFile::ReadOnly ) )
		{
			while ( !includedFile.atEnd() )
			{
				QByteArray line = includedFile.readLine().trimmed();
				QString includedKey = QString( line );
				if ( includedKey != "" )
				{
					mIncludedKeys.push_back( includedKey );
				}
			}

			includedFile.close();
		}
	}

	initialize( mFDB, mLDB, mLabelName );
}

//-----------------------------------------------------------------------------

DataPackage::DataPackage( muw::TabularData& aFDB, muw::TabularData& aLDB, QString aLabelName, QStringList aIncludedKeys )
{
	mIncludedKeys = aIncludedKeys;
	initialize( aFDB, aLDB, aLabelName );
}

//-----------------------------------------------------------------------------

void DataPackage::initialize( muw::TabularData& aFDB, muw::TabularData& aLDB, QString aLabelName )
{
	mFDB = aFDB;
	mLDB = aLDB;
	mLabelName = aLabelName;

	muw::TabularDataFilter filter;
	
	if ( !mIncludedKeys.isEmpty() )
	{
		mFDB = filter.subTableByKeys( mFDB, mIncludedKeys );
		mLDB = filter.subTableByKeys( mLDB, mIncludedKeys );
	}

	filter.eraseIncompleteRecords( mFDB );

	mFeatureCount = mFDB.columnCount();

	QStringList labelNames = mLDB.headerNames();
	mLabelIndex = labelNames.indexOf( mLabelName );

	if ( mLabelIndex == -1 )
	{
		//qDebug() << "DataPackage ERROR: Label name " << mLabelName << "is not part of label names: " << labelNames;
		mIsValidDataset = false;
		return;
	}

	mLabelOutcomes = filter.labelGroups( mLDB, mLabelIndex );
	mSampleKeys = filter.commonKeys( mFDB, mLDB, mLabelIndex );

	if ( mLabelOutcomes.size() < 2 )
	{
		qDebug() << "DataPackage ERROR: Insufficient number of label outcomes: " << mLabelOutcomes;
		//system( "pause" );
		mIsValidDataset = false;
		return;
	}


	mIsValidDataset = true;
}

//-----------------------------------------------------------------------------

DataPackage::~DataPackage()
{
	mFDB.clear();
	mLDB.clear();
}

//-----------------------------------------------------------------------------

}