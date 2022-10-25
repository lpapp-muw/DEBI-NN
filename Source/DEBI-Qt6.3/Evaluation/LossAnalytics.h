/*!
* \file
* This file is part of the Evaluation module.
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
#include <Evaluation/AbstractAnalytics.h>
#include <DataRepresentation/TabularData.h>

namespace muw
{

//-----------------------------------------------------------------------------

/*!
* \brief The LossAnalytics class is an inherited class from AbstractAnalytics. Manifests analytics based on loss calculations.
*
* \details
*/

class Evaluation_API LossAnalytics : public AbstractAnalytics
{

public:

	/*!
	\brief Constructor.
	\param [in] aDataPackage the data package to be used in order to evaluate a model.
	\param [in] aUnit the unit of the evaluation.
	*/
	LossAnalytics( muw::DataPackage* aDataPackage, QString aUnit, double aAlpha );

	/*!
	\brief Destructor.
	*/
	~LossAnalytics();

	/*!
	\brief Evaluates the given model by the predefined unit type. Currently, the only implemented option is entropy loss.
	\param [in] aModel the model to evaluate.
	*/
	double evaluate( muw::AbstractModel* aModel ) override;

	/*!
	\brief Returns all the implemented unit measures of LossAnalytics of the model.
	*/
	QMap< QString, double > allValues() override;


private:

	/*!
	\brief Default constructor cannot be used.
	*/
	LossAnalytics();

private:

	double  mEntropyLoss;  //!< The calculated entropy loss value.
	double  mAlpha;        //!< The alpha value which weighs correct and incorrect (1-alpha) predictions.
	QMap< QString, double >  mLabelRatios;
};

//-----------------------------------------------------------------------------

}
