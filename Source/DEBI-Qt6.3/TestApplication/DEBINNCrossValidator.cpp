/*!
* \file
* This file is part of TestApplication module.
* Member function definitions for DEBINNCrossValidator class.
*
* \remarks
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#include "DEBINNCrossValidator.h"
#include <FileIo/TabularDataFileIo.h>
#include <Evaluation/DataPackage.h>
#include <Evaluation/DEBINNFactory.h>
#include <Evaluation/LossAnalytics.h>
#include <Evaluation/ConfusionMatrixAnalytics.h>
#include <QDebug>
#include <QDir>
#include <random>

namespace muw
{

//-----------------------------------------------------------------------------

DEBINNCrossValidator::DEBINNCrossValidator( QString aExecutionFolderPath )
:
	mExecutionFolderPath( aExecutionFolderPath ),
	//mDEBINNFactorySettingFolderPath( aDEBINNFactorySettingFolderPath ),
	mSettingsVariants(),
	mCrossValidationResults(),
	mFoldCount( 0 ),
	mLogTable(),
	mResultTable()
{
	QStringList filters = { "*.ini*" };
	QStringList settingsList = QDir( mExecutionFolderPath + "/Settings/" ).entryList( filters );

	for ( auto settingPath : settingsList )
	{
		QString settingFileName = settingPath.split("/").last();
		mSettingsVariants.push_back( new QSettings( mExecutionFolderPath + "/Settings/" + settingPath, QSettings::IniFormat ) );
	}

	QDir dir;
	if ( !dir.exists( mExecutionFolderPath + "/Log/" ) ) dir.mkpath( mExecutionFolderPath + "/Log/" );
}

//-----------------------------------------------------------------------------

DEBINNCrossValidator::~DEBINNCrossValidator()
{
	for ( auto setting : mSettingsVariants )
	{
		delete setting;
		setting = nullptr;
	}

	mSettingsVariants.clear();

	mLogTable.clear();
	mResultTable.clear();
}

//-----------------------------------------------------------------------------

void DEBINNCrossValidator::execute()
{
	QStringList filters = { "*Fold*" };
	QStringList foldList = QDir( mExecutionFolderPath + "/Dataset/" ).entryList( filters );

	mFoldCount = foldList.size();

	if ( mFoldCount == 0 )
	{
		qDebug() << "Warning: No Fold-... subfolders detected under" << mExecutionFolderPath + "/Dataset/";
	}

	for ( auto fold : foldList )
	{
		executeFold( mExecutionFolderPath + "/Dataset/" + fold + "/" );
	}

	qDebug() << "Processing" << mExecutionFolderPath << "finished.";
}

//-----------------------------------------------------------------------------

void DEBINNCrossValidator::executeFold( QString aFoldFolderPath )
{
	muw::TabularDataFileIo FIO;
	muw::TabularData TrainF;
	muw::TabularData TrainL;
	muw::TabularData ValidateF;
	muw::TabularData ValidateL;
	muw::TabularData TestF;
	muw::TabularData TestL;

	bool isTrainSet    = false;
	bool isValidateSet = false;
	bool isTestSet     = false;


	if ( QFile::exists( aFoldFolderPath + "/TrainF.csv" ) )
	{
		FIO.load( aFoldFolderPath + "/TrainF.csv", TrainF );
		FIO.load( aFoldFolderPath + "/TrainL.csv", TrainL );
		isTrainSet = true;
	}
	else
	{
		qDebug() << "Error: No training subset found in the given fold" << aFoldFolderPath;
		return;
	}

	if ( QFile::exists( aFoldFolderPath + "/ValidateF.csv" ) )
	{
		FIO.load( aFoldFolderPath + "/ValidateF.csv", ValidateF );
		FIO.load( aFoldFolderPath + "/ValidateL.csv", ValidateL );
		isValidateSet = true;
	}
	else
	{
		qDebug() << "Note: No validation subset found in the given fold" << aFoldFolderPath << "- Training subset will be further splitted to train-validate subsets.";
	}

	if ( QFile::exists( aFoldFolderPath + "/TestF.csv" ) )
	{
		FIO.load( aFoldFolderPath + "/TestF.csv", TestF );
		FIO.load( aFoldFolderPath + "/TestL.csv", TestL );
		isTestSet = true;
	}
	else
	{
		qDebug() << "Error: No test subset found in the given fold" << aFoldFolderPath << "- Cross-validation is not possible to perform.";
		return;
	}

	QString labelName = TrainL.headerNames().last();

	muw::DataPackage DPTrain(    TrainF,    TrainL,    labelName );
	muw::DataPackage DPValidate( ValidateF, ValidateL, labelName );
	muw::DataPackage DPTest(     TestF,     TestL,     labelName );

	for ( int h = 0; h < mSettingsVariants.size(); ++h )
	{
		QSettings* currentSettings = mSettingsVariants.at( h );
		QString settingsLogKey = currentSettings->fileName().split("/").last();

		qDebug() << "SETTINGS:" << settingsLogKey;
		QString foldCount = aFoldFolderPath.split( "Fold-" ).at( 1 ).split( "/" ).at( 0 );

		double alpha = currentSettings->value( "Analytics/Loss/Alpha" ).toDouble();
		if ( alpha < DBL_EPSILON )
		{
			qDebug() << "Loss Alpha (from settings):" << alpha;
			alpha = 0.9;
		}
		qDebug() << "Loss Alpha:" << alpha;

		std::shared_ptr< muw::DEBINN > innerSNN;
		muw::DEBINNFactory innerFactory( currentSettings );

		if ( isValidateSet )
		{
			innerSNN = innerFactory.generate( &DPTrain, &DPValidate );
		}
		else
		{
			innerSNN = innerFactory.generate( &DPTrain, nullptr );
		}

		auto log = innerFactory.log();

		if ( mLogTable.headerNames().isEmpty() )
		{
			QStringList logHeader = log.value( "Iteration" ).toStringList();
			mLogTable.setHeader( logHeader );
			auto mutationLog = log.value( "MutationRate" ).toList();
			mLogTable.insert( settingsLogKey + "/" + "MutationRate", mutationLog );
		}

		if (mResultTable.headerNames().isEmpty())
		{
			QStringList resultHeader;
			resultHeader.append( { "SNS", "SPC", "PPV", "NPV", "ACC", "EntropyLoss"});
			mResultTable.setHeader( resultHeader );
		}

		auto trainingLog = log.value( "TrainingError" ).toList();
		mLogTable.insert( settingsLogKey + "/" + "TrainingError/" + "Fold-" + foldCount, trainingLog );

		auto validateLog = log.value( "ValidateError" ).toList();
		if ( !validateLog.isEmpty() ) mLogTable.insert( settingsLogKey + "/" + "ValidateError/" + "Fold-" + foldCount, validateLog );

		muw::ConfusionMatrixAnalytics CATrain( &DPTrain, "ROCDistance" );
		muw::LossAnalytics LATrain(            &DPTrain, "EntropyLoss", alpha );
		muw::ConfusionMatrixAnalytics CATest( &DPTest, "ROCDistance" );
		muw::LossAnalytics LATest(            &DPTest, "EntropyLoss", alpha );

		CATrain.evaluate( innerSNN.get() );
		LATrain.evaluate( innerSNN.get() );
		CATest.evaluate( innerSNN.get() );
		LATest.evaluate( innerSNN.get() );

		auto LATrainAllValues = LATrain.allValues();
		auto CATrainAllValues = CATrain.allValues();
		auto LATestAllValues  = LATest.allValues();
		auto CATestAllValues  = CATest.allValues();

		qDebug() << "TRAINING performance for setting" << settingsLogKey;		
		qDebug() << "Accuracy                          " << CATrainAllValues.value("ACC");
		qDebug() << "Ballanced Accuracy                " << CATrainAllValues.value("BACC");
		qDebug() << "Sensitivity                       " << CATrainAllValues.value("SNS");
		qDebug() << "Specificity                       " << CATrainAllValues.value("SPC");
		qDebug() << "Positive Predictive Value         " << CATrainAllValues.value("PPV");
		qDebug() << "Negative Predictive Value         " << CATrainAllValues.value("NPV");
		qDebug() << "Matthew's Correlation Coefficient " << CATrainAllValues.value("MCC");
		qDebug() << "ROC Distance                      " << CATrainAllValues.value("ROCd");
		qDebug() << "F0.5-Score                        " << CATrainAllValues.value("F0.5-Score");
		qDebug() << "F2-Score                          " << CATrainAllValues.value("F2-Score");
		qDebug() << "Entropy Loss                      " << LATrainAllValues.value("EntropyLoss");
		qDebug() << Qt::endl;

		qDebug() << "TEST performance for setting" << settingsLogKey;
		qDebug() << "Accuracy                          " << CATestAllValues.value("ACC");
		qDebug() << "Ballanced Accuracy                " << CATestAllValues.value("BACC");
		qDebug() << "Sensitivity                       " << CATestAllValues.value("SNS");
		qDebug() << "Specificity                       " << CATestAllValues.value("SPC");
		qDebug() << "Positive Predictive Value         " << CATestAllValues.value("PPV");
		qDebug() << "Negative Predictive Value         " << CATestAllValues.value("NPV");
		qDebug() << "Matthew's Correlation Coefficient " << CATestAllValues.value("MCC");
		qDebug() << "ROC Distance                      " << CATestAllValues.value("ROCd");
		qDebug() << "F0.5-Score                        " << CATestAllValues.value("F0.5-Score");
		qDebug() << "F2-Score                          " << CATestAllValues.value("F2-Score");
		qDebug() << "Entropy Loss                      " << LATestAllValues.value("EntropyLoss");
		qDebug() << Qt::endl;

		QVariantList resultLogRecord;
		resultLogRecord.push_back( CATestAllValues.value( "SNS" ) );
		resultLogRecord.push_back( CATestAllValues.value( "SPC" ) );
		resultLogRecord.push_back( CATestAllValues.value( "PPV" ) );
		resultLogRecord.push_back( CATestAllValues.value( "NPV" ) );
		resultLogRecord.push_back( CATestAllValues.value( "ACC" ) );
		resultLogRecord.push_back( LATestAllValues.value( "EntropyLoss" ) );

		mResultTable.insert( settingsLogKey + "/" + "TestError/" + "Fold-" + foldCount, resultLogRecord );

		qDebug() << "----" << Qt::endl;
	}
	
	muw::TabularDataFileIo logFIO;
	logFIO.save( mExecutionFolderPath + "/Log/DetailedLog.csv", mLogTable );
	logFIO.save( mExecutionFolderPath + "/Log/ResultLog.csv", mResultTable );

}

//-----------------------------------------------------------------------------

}
