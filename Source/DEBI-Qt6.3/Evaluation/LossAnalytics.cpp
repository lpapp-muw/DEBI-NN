/*!
* \file
* Member class definitions of LossAnalytics. This file is part of the Evaluation module.
*
* \remarks
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#pragma once

#include <Evaluation/LossAnalytics.h>
#include <Evaluation/DEBINN.h>
#include <Evaluation/TabularDataFilter.h>
#include <QDebug>

namespace muw
{

//-----------------------------------------------------------------------------

LossAnalytics::LossAnalytics( muw::DataPackage* aDataPackage, QString aUnit, double aAlpha )
:
	AbstractAnalytics( aDataPackage, aUnit ),
	mEntropyLoss( 0.0 ),
	mAlpha( aAlpha ),
	mLabelRatios()
{
	TabularDataFilter filter;
	auto labelOutcomes = mDataPackage->labelOutcomes();
	
	for ( int i = 0; i < labelOutcomes.size(); ++i )
	{
		auto labelGroups = filter.keysByLabelGroup( aDataPackage->labelDB(), aDataPackage->labelIndex(), labelOutcomes.at( i ) );
		mLabelRatios[ labelOutcomes.at( i ) ] = double( labelOutcomes.size() ) * ( 1.0 - ( double( labelGroups.size() ) / double( mDataPackage->sampleKeys().size() ) ) );
	}
}

//-----------------------------------------------------------------------------

LossAnalytics::~LossAnalytics()
{
	mLabelRatios.clear();
}

//-----------------------------------------------------------------------------

double LossAnalytics::evaluate( muw::AbstractModel* aModel )
{
	if ( mUnit == "EntropyLoss" )
	{
		mEntropyLoss = 0.0;

		auto sampleKeys = mDataPackage->sampleKeys();
		for ( auto sampleKey : sampleKeys )
		{
			auto featureVector = mDataPackage->featureDB().value( sampleKey );

			QVector< double > featureVectorDouble;
			for ( auto feature : featureVector )
			{
				featureVectorDouble.push_back( feature.toDouble() );
			}

			QVariantMap prediction = aModel->evaluate( featureVectorDouble ).toMap();
			QString originalLabel  = mDataPackage->labelDB().valueAt( sampleKey, mDataPackage->labelIndex() ).toString();
			int originalIndex      = mDataPackage->labelOutcomes().indexOf( originalLabel );

			auto predictionKeys = prediction.keys();
			for ( int p = 0; p < predictionKeys.size(); ++p )
			{
				auto evaluatedLabel = predictionKeys.at( p );
				int evaluatedIndex  = mDataPackage->labelOutcomes().indexOf( evaluatedLabel );

				double beta = mLabelRatios.value( evaluatedLabel );

				double yi    = evaluatedIndex == originalIndex ? mAlpha : 1.0 - mAlpha;
				double yiHat = prediction.value( evaluatedLabel ).toDouble();

				if ( yi < DBL_EPSILON ) continue;  // Skip calculation.

				double logTerm = 0.0;

				if ( yiHat < DBL_EPSILON )
				{
					logTerm = -20;  // DBL_EPSILON = 10^-16 --> log( DBL_EPSILON ) > 20.
				}
				else if ( std::abs( yiHat - 1.0 ) < DBL_EPSILON  )  // No need to calculate log( 1.0 ) --> 0.0 --> no need to accumulate current loss either.
				{
					continue;
				}
				else
				{
					logTerm = std::log( yiHat );
				}

				mEntropyLoss -= ( yi * logTerm * beta );
			}
		}

		//double additionalLoss = 0.0;  // This is a placeholder for L1, L2, etc loss...
		//muw::DEBINN* DEBIN = dynamic_cast< muw::DEBINN* >( aModel );
		//if ( DEBIN != nullptr )
		//{
		//	
		//	 //TODO: This must be an input or hypermarameter!
		//	
		//	/*double lambda = 0.05;
		//	auto l1Norm = DEBIN->l1Norm() * lambda;
		//	additionalLoss = l1Norm;*/
		//}

		if ( mEntropyLoss < DBL_EPSILON ) return 0.0;

		double entropyLoss = mEntropyLoss  / sampleKeys.size();

		if ( std::isnan( entropyLoss ) )
		{
			qDebug() << "WARNING: entropy loss is" << entropyLoss;
			qDebug() << "         mEntropyLoss:  " << mEntropyLoss;
			qDebug() << "         sample count:  " << sampleKeys.size();
		}

		return entropyLoss;
	}
	else
	{
		qDebug() << "EntropyLoss - ERROR: Unknown unit" << mUnit;
		return std::numeric_limits< double >::quiet_NaN();
	}
}

//-----------------------------------------------------------------------------

QMap< QString, double > LossAnalytics::allValues()
{
	QMap< QString, double > result;
	result[ "EntropyLoss" ] = mEntropyLoss / mDataPackage->sampleKeys().size();

	return result; 
}

//-----------------------------------------------------------------------------

}
