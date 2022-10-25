/*!
* \file
* Member class definitions of ConfusionMatrixAnalytics. This file is part of the Evaluation module.
*
* \remarks
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#include <Evaluation/ConfusionMatrixAnalytics.h>
#include <Evaluation/TabularDataFilter.h>
#include <FileIo/TabularDataFileIo.h>
#include <cfloat>
#include <QDebug>

namespace muw
{

//-----------------------------------------------------------------------------

ConfusionMatrixAnalytics::ConfusionMatrixAnalytics( muw::DataPackage* aDataPackage, QString aUnit )
:
	AbstractAnalytics( aDataPackage, aUnit ),
	mConfusionMatrix( nullptr ),
	mConfusionMatrixMeasure( ConfusionMatrixMeasure::ROC ),
	mTPs(),
	mTNs(),
	mFPs(),
	mFNs(),
	mDiagonalSum( 0.0 ),
	mSum( 0.0 ),
	mIsValid( false )
{
	if ( aUnit == "ROCDistance" )
	{
		mConfusionMatrixMeasure = ConfusionMatrixMeasure::ROC;
	}
	else if ( aUnit == "ACC" )
	{
		mConfusionMatrixMeasure = ConfusionMatrixMeasure::ACC;
	}
	else if ( aUnit == "FScore" )
	{
		mConfusionMatrixMeasure = ConfusionMatrixMeasure::FScore;
	}
	else if ( aUnit == "SNS" )
	{
		mConfusionMatrixMeasure = ConfusionMatrixMeasure::SNS;
	}
	else if ( aUnit == "SPC" )
	{
		mConfusionMatrixMeasure = ConfusionMatrixMeasure::SPC;
	}
	else if ( aUnit == "PPV" )
	{
		mConfusionMatrixMeasure = ConfusionMatrixMeasure::PPV;
	}
	else if ( aUnit == "NPV" )
	{
		mConfusionMatrixMeasure = ConfusionMatrixMeasure::NPV;
	}
	else if ( aUnit == "MCC" )
	{
		mConfusionMatrixMeasure = ConfusionMatrixMeasure::MCC;
	}
	else if ( aUnit == "BACC" )
	{
		mConfusionMatrixMeasure = ConfusionMatrixMeasure::BACC;
	}
}

//-----------------------------------------------------------------------------

ConfusionMatrixAnalytics::~ConfusionMatrixAnalytics()
{
	delete mConfusionMatrix;
	mConfusionMatrix = nullptr;

	mTPs.clear();
	mTNs.clear();
	mFPs.clear();
	mFNs.clear();
}

//-----------------------------------------------------------------------------

void ConfusionMatrixAnalytics::resetConfusionMatrix()
{
	delete mConfusionMatrix;
	mConfusionMatrix = new muw::Array2D< unsigned int >( mDataPackage->labelOutcomes().size(), mDataPackage->labelOutcomes().size() );
	mIsValid = false;

}

//-----------------------------------------------------------------------------

void ConfusionMatrixAnalytics::resetContainers()
{
	mTPs.clear();
	mTNs.clear();
	mFPs.clear();
	mFNs.clear();

	mDiagonalSum = 0.0;
	mSum = 0.0;
}

//-----------------------------------------------------------------------------

double ConfusionMatrixAnalytics::evaluate( muw::AbstractModel* aModel )
{
	if ( !mDataPackage->isValid() )
	{
		qDebug() << "ConfusionMatrixAnalytics ERROR: Invalid dataset given, unable to evaluate";
		system( "pause" );
		return -DBL_MAX;
	}

	// Reset the confusion matrix.
	resetConfusionMatrix();

	muw::TabularData FDB = mDataPackage->featureDB();
	muw::TabularData LDB = mDataPackage->labelDB();

	// Evaluate the FDB datasets, fill in the confusion matrix.
	for ( auto key : mDataPackage->sampleKeys() )
	{
		QVariantList featureVectorVariant = FDB.value( key );
		QVariantList labelVariant         = LDB.value( key );

		// TODO: Filter out here only those features that are known by the Model.
		QVector< double > featureVector;
		for ( int i = 0; i < featureVectorVariant.size(); ++i )
		{
			featureVector.push_back( featureVectorVariant.at( i ).toDouble() );
		}

		QVariantMap prediction = aModel->evaluate( featureVector ).toMap();
		QString originalLabel  = mDataPackage->labelDB().valueAt( key, mDataPackage->labelIndex() ).toString();

		double maxValue = -DBL_MAX;
		QString evaluatedLabel;
		auto predictionKeys = prediction.keys();
		for ( int p = 0; p < predictionKeys.size(); ++p )
		{
			auto label = predictionKeys.at( p );
			auto probability = prediction.value( label ).toDouble();
			if ( probability > maxValue )
			{
				maxValue = probability;
				evaluatedLabel = label;
			}
		}

		int originalIndex  = mDataPackage->labelOutcomes().indexOf( originalLabel );
		int evaluatedIndex = mDataPackage->labelOutcomes().indexOf( evaluatedLabel );

		mConfusionMatrix->addEntry( evaluatedIndex, originalIndex );
	}

	mIsValid = true;

	calculateAtomicErrors();

	switch ( mConfusionMatrixMeasure )
	{
		case ConfusionMatrixMeasure::ROC:
		{
			return rocDistance();
			break;
		}
		case ConfusionMatrixMeasure::BACC:
		{
			return 1.0 - bacc();
			break;
		}
		case ConfusionMatrixMeasure::FScore:
		{
			return 1.0 - fScore( 2.0 );
			break;
		}
		case ConfusionMatrixMeasure::ACC:
		{
			return 1.0 - acc();
			break;
		}
		case ConfusionMatrixMeasure::SNS:
		{
			return 1.0 - sns();
			break;
		}
		case ConfusionMatrixMeasure::SPC:
		{
			return 1.0 - spc();
			break;
		}
		case ConfusionMatrixMeasure::PPV:
		{
			return 1.0 - ppv();
			break;
		}
		case ConfusionMatrixMeasure::NPV:
		{
			return 1.0 - npv();
			break;
		}
		case ConfusionMatrixMeasure::MCC:
		{
			return mcc();
			break;
		}
		default:
		{
			return rocDistance();
			break;
		}
	}
}

//-----------------------------------------------------------------------------

void ConfusionMatrixAnalytics::calculateAtomicErrors()
{
	resetContainers();

	// Binary classifier.
	if ( mConfusionMatrix->rowCount() == 2 )
	{
		mTNs.push_back( double( mConfusionMatrix->at( 0, 0 ) ) );
		mFNs.push_back( double( mConfusionMatrix->at( 0, 1 ) ) );
		mFPs.push_back( double( mConfusionMatrix->at( 1, 0 ) ) );
		mTPs.push_back( double( mConfusionMatrix->at( 1, 1 ) ) );
		mDiagonalSum = mTNs.at( 0 ) + mTPs.at( 0 );
		mSum = mTNs.at( 0 ) + mFNs.at( 0 ) + mFPs.at( 0 ) + mTPs.at( 0 );
	}
	else  // Multi-class classifier.
	{
		double geomMeanRocDistances = 0.0;

		for ( unsigned int rowIndex = 0; rowIndex < mConfusionMatrix->rowCount(); ++rowIndex )
		{
			mDiagonalSum += mConfusionMatrix->at( rowIndex, rowIndex );
		}

		for ( unsigned int rowIndex = 0; rowIndex < mConfusionMatrix->rowCount(); ++rowIndex )
		{
			double rowSum = 0.0;
			double columnSum = 0.0;
			double TP = 0.0;
			double FP = 0.0;
			double FN = 0.0;
			double TN = 0.0;

			//Calculate row and column sums.
			for ( unsigned int columnIndex = 0; columnIndex < mConfusionMatrix->columnCount(); ++columnIndex )
			{
				rowSum    += double( mConfusionMatrix->at( rowIndex, columnIndex ) );  //Sum row (FP + TP)
				columnSum += double( mConfusionMatrix->at( columnIndex, rowIndex ) );  //Sum column (FN + TP)
				mSum      += double( mConfusionMatrix->at( rowIndex, columnIndex ) );
			}

			TP = double( mConfusionMatrix->at( rowIndex, rowIndex ) );  //TP is the actual diagonal element.
			FP = rowSum - TP;  //Row contains TP too.
			FN = columnSum - TP;  //Column contains TP too.
			TN = mDiagonalSum - TP;  //Diagonal sum contains TP too.

			mTPs.push_back( TP );
			mFPs.push_back( FP );
			mFNs.push_back( FN );
			mTNs.push_back( TN );
		}
	}
}

//-----------------------------------------------------------------------------

double ConfusionMatrixAnalytics::rocDistance()
{
	// Binary classifier.
	if ( mConfusionMatrix->rowCount() == 2 )
	{
		double TN = mTNs.at( 0 );
		double FN = mFNs.at( 0 );
		double FP = mFPs.at( 0 );
		double TP = mTPs.at( 0 );

		double TPR = TP / ( TP + FN );  //True Positive Rate.
		double FPR = FP / ( FP + TN );  //False Positive Rate.

		return sqrt( ( ( 1.0 - TPR ) * ( 1.0 - TPR ) ) + ( FPR *  FPR ) );
	}
	else  // Multi-class classifier.
	{
		double diagonalSum = 0.0;
		double totalSum = 0.0;
		double geomMeanRocDistances = 0.0;

		for ( unsigned int rowIndex = 0; rowIndex < mConfusionMatrix->rowCount(); ++rowIndex )
		{
			double TN = mTNs.at( rowIndex );
			double FN = mFNs.at( rowIndex );
			double FP = mFPs.at( rowIndex );
			double TP = mTPs.at( rowIndex );

			double TPR = TP / ( TP + FN );  //True Positive Rate.
			double FPR = FP / ( FP + TN );  //False Positive Rate.

			double actualRocDistance = sqrt( ( ( 1.0 - TPR ) * ( 1.0 - TPR ) ) + ( FPR *  FPR ) );

			geomMeanRocDistances += actualRocDistance * actualRocDistance;
		}

		return sqrt( geomMeanRocDistances );
	}
}

//-----------------------------------------------------------------------------

double ConfusionMatrixAnalytics::fScore( double aBeta )
{

	if ( !mIsValid ) return NAN;

	double betaSqr = std::pow( aBeta, 2.0 );

	// Binary classifier.
	if ( mConfusionMatrix->rowCount() == 2 )
	{
		double TN = mTNs.at( 0 );
		double FN = mFNs.at( 0 );
		double FP = mFPs.at( 0 );
		double TP = mTPs.at( 0 );

		return ( ( 1.0 + betaSqr ) * TP ) / ( ( ( 1.0 + betaSqr ) * TP ) + ( betaSqr * FN ) + FP );
	}
	else  // Multi-class classifier.
	{
		double geomMeanFScores = 0.0;

		for ( unsigned int rowIndex = 0; rowIndex < mConfusionMatrix->rowCount(); ++rowIndex )
		{
			double TN = mTNs.at( rowIndex );
			double FN = mFNs.at( rowIndex );
			double FP = mFPs.at( rowIndex );
			double TP = mTPs.at( rowIndex );

			double actualFScore = ( ( 1.0 + betaSqr ) * TP ) / ( ( ( 1.0 + betaSqr ) * TP ) + ( betaSqr * FN ) + FP );
			geomMeanFScores += actualFScore * actualFScore;
		}

		return sqrt( geomMeanFScores );
	}
}

//-----------------------------------------------------------------------------

double ConfusionMatrixAnalytics::acc()
{
	if ( !mIsValid ) return NAN;

	return mDiagonalSum / mSum;
}

//-----------------------------------------------------------------------------

double ConfusionMatrixAnalytics::bacc()
{
	if ( !mIsValid ) return NAN;

	return ( sns() + spc() ) / 2.0;
}

//-----------------------------------------------------------------------------

double ConfusionMatrixAnalytics::sns()
{
	if ( !mIsValid ) return NAN;

	double FN = 0.0;
	double TP = 0.0;

	// Binary classifier.
	if ( mConfusionMatrix->rowCount() == 2 )
	{
		FN = mFNs.at( 0 );
		TP = mTPs.at( 0 );
	}
	else  // Multi-class classifier.
	{
		for ( unsigned int rowIndex = 0; rowIndex < mConfusionMatrix->rowCount(); ++rowIndex )
		{
			FN += mFNs.at( rowIndex );
			TP += mTPs.at( rowIndex );
		}
	}

	return TP / ( TP + FN );
}

//-----------------------------------------------------------------------------

double ConfusionMatrixAnalytics::spc()
{
	if ( !mIsValid ) return NAN;

	double TN = 0.0;
	double FP = 0.0;

	// Binary classifier.
	if ( mConfusionMatrix->rowCount() == 2 )
	{
		TN = mTNs.at( 0 );
		FP = mFPs.at( 0 );
	}
	else  // Multi-class classifier.
	{
		for ( unsigned int rowIndex = 0; rowIndex < mConfusionMatrix->rowCount(); ++rowIndex )
		{
			TN += mTNs.at( rowIndex );
			FP += mFPs.at( rowIndex );
		}
	}

	return TN / ( TN + FP );
}

//-----------------------------------------------------------------------------

double ConfusionMatrixAnalytics::ppv()
{
	if ( !mIsValid ) return NAN;

	double FP = 0.0;
	double TP = 0.0;

	// Binary classifier.
	if ( mConfusionMatrix->rowCount() == 2 )
	{
		FP = mFPs.at( 0 );
		TP = mTPs.at( 0 );
	}
	else  // Multi-class classifier.
	{
		for ( unsigned int rowIndex = 0; rowIndex < mConfusionMatrix->rowCount(); ++rowIndex )
		{
			FP += mFPs.at( rowIndex );
			TP += mTPs.at( rowIndex );
		}
	}

	return TP / ( TP + FP + DBL_EPSILON );
}

//-----------------------------------------------------------------------------

double ConfusionMatrixAnalytics::npv()
{
	if ( !mIsValid ) return NAN;

	double TN = 0.0;
	double FN = 0.0;

	// Binary classifier.
	if ( mConfusionMatrix->rowCount() == 2 )
	{
		TN = mTNs.at( 0 );
		FN = mFNs.at( 0 );
	}
	else  // Multi-class classifier.
	{
		for ( unsigned int rowIndex = 0; rowIndex < mConfusionMatrix->rowCount(); ++rowIndex )
		{
			TN += mTNs.at( rowIndex );
			FN += mFNs.at( rowIndex );
		}
	}

	return TN / ( TN + FN + DBL_EPSILON );
}

//-----------------------------------------------------------------------------

double ConfusionMatrixAnalytics::mcc()
{
	if ( !mIsValid ) return NAN;

	double TN = 0.0;
	double FN = 0.0;
	double FP = 0.0;
	double TP = 0.0;

	// Binary classifier.
	if ( mConfusionMatrix->rowCount() == 2 )
	{
		TN = mTNs.at( 0 );
		FN = mFNs.at( 0 );
		FP = mFPs.at( 0 );
		TP = mTPs.at( 0 );
	}
	else  // Multi-class classifier.
	{
		for ( unsigned int rowIndex = 0; rowIndex < mConfusionMatrix->rowCount(); ++rowIndex )
		{
			TN += mTNs.at( rowIndex );
			FN += mFNs.at( rowIndex );
			FP += mFPs.at( rowIndex );
			TP += mTPs.at( rowIndex );
		}
	}

	return ( ( TP * TN ) - ( FP * FN ) ) / std::sqrt( ( TP + FP )*( TP + FN )*( TN + FP )*( TN + FN ) );
}

//-----------------------------------------------------------------------------

QMap< QString, double > ConfusionMatrixAnalytics::allValues()
{
	QMap< QString, double > values;
	values.insert( "ACC", acc() );
	values.insert( "BACC", bacc() );
	values.insert( "SNS", sns() );
	values.insert( "SPC", spc() );
	values.insert( "PPV", ppv() );
	values.insert( "NPV", npv() );
	values.insert( "MCC", mcc() );
	values.insert( "ROCd", rocDistance() );
	values.insert( "F0.5-Score", fScore( 0.5 ) );
	values.insert( "F2-Score", fScore( 2.0 ) );

	return values;
}

//-----------------------------------------------------------------------------

}
