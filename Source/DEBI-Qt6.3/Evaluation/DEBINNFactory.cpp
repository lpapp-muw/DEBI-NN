/*!
* \file
* Member class definitions of DEBINNFactory. This file is part of the Evaluation module.
*
* \remarks
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#include <Evaluation/DEBINNFactory.h>
#include <Evaluation/GeneticAlgorithmOptimizer.h>
#include <QFile>
#include <QDebug>

namespace muw
{

//-----------------------------------------------------------------------------

DEBINNFactory::DEBINNFactory()
:
	mSettings(),
	mDataPackage( nullptr ),
	mLog()
{
}


//-----------------------------------------------------------------------------

DEBINNFactory::DEBINNFactory( QSettings* aSettings )
:
	mSettings(),
	mDataPackage( nullptr ),
	mLog()
{
	initialize( aSettings );
}

//-----------------------------------------------------------------------------

DEBINNFactory::DEBINNFactory( QString aSettingsFolderPath )
:
	mSettings(),
	mDataPackage( nullptr ),
	mLog()
{

	QString settingFilePath = aSettingsFolderPath + "/settings.ini";

	if ( QFile::exists( settingFilePath ) )
	{
		QSettings settings( settingFilePath, QSettings::IniFormat );
		initialize( &settings );
	}
	else
	{
		qDebug() << "DEBINNFactory - WARNING: Invalid path" << settingFilePath;
	}
}

//-----------------------------------------------------------------------------

DEBINNFactory::~DEBINNFactory()
{
	delete mDataPackage;
	mDataPackage = nullptr;

	mLog.clear();
}

//-----------------------------------------------------------------------------

std::shared_ptr< muw::DEBINN > DEBINNFactory::generate( muw::DataPackage * aDataPackage, muw::DataPackage* aValidationDataPackage )
{
	std::shared_ptr< muw::DEBINN > SNN;

	if ( mSettings.isEmpty() )
	{
		qDebug() << "DEBINNFactory - ERROR: mSettings is empty!";
		return SNN;
	}

	if ( aDataPackage == nullptr )
	{
		qDebug() << "DEBINNFactory - ERROR: DataPackage is nullptr.";
		return SNN;
	}

	auto currentSettings = mSettings;

	currentSettings.insert( "Model/Layers/Input/NeuronCount",  aDataPackage->featureDB().columnCount() );
	currentSettings.insert( "Model/Layers/Output/NeuronCount", aDataPackage->labelOutcomes().size() );
	currentSettings.insert( "Data/PredictionLabels",           aDataPackage->labelOutcomes() );

	SNN = std::make_shared< muw::DEBINN >( currentSettings );
	QVector< double > optimalGeneSequence;

	muw::GeneticAlgorithmOptimizer GA( SNN.get(), aDataPackage, currentSettings, aValidationDataPackage );
	GA.build();

	mLog = GA.log();

	optimalGeneSequence = GA.result();
	
	SNN->set( optimalGeneSequence );
	
	return SNN;
}

//-----------------------------------------------------------------------------

std::shared_ptr<muw::DEBINN> DEBINNFactory::generate( QString aDEBIModelFolderPath )
{

	muw::DEBINN SNN( mSettings );
	SNN.load( aDEBIModelFolderPath );

	return std::make_shared< muw::DEBINN >( SNN );
}

//-----------------------------------------------------------------------------

void DEBINNFactory::initialize( QSettings * aSettings )
{
	if ( aSettings != nullptr )
	{
		mSettings.clear();

		QStringList keys = aSettings->allKeys();
		for ( auto key : keys )
		{
			auto value = aSettings->value( key );
			mSettings.insert( key, value );
		}
	}
	else
	{
		qDebug() << "DEBINNFactory - WARNING: settings is nullptr in initialize(). Request ignored.";
	}
}

//-----------------------------------------------------------------------------

}
