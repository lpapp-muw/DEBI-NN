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
#include <DataRepresentation/Array2D.h>
#include <DataRepresentation/TabularData.h>

namespace muw
{

/*!
\brief The ConfusionMatrixMeasure enum is to store the unit type of the confusion matrix to be used for determining the error of the given model.
*/
enum class ConfusionMatrixMeasure
{
	ROC = 0,
	FScore,
	ACC,
	SNS,
	SPC,
	PPV,
	NPV,
	MCC,
	BACC
};

//-----------------------------------------------------------------------------

/*!
* \brief The ConfusionMatrixAnalytics class is an inherited class from AbstractAnalytics. Manifests analytics based on confusion matrix calculations.
*
* \details ConfusionMatrixAnalytics class First generates a binary or multi-class confusion matrix based on the input DataPackage and the AbstractModel to evaluate - based on the classification characteristics of the model.
* Confusion matrix values such as sensitivity, specificity, positive and negative predictive value as well as accuracy, FScore, ROC distance and balanced accuracy can be calculated.
* The evaluate() function of ConfusionMatrixAnalytics returns with a classification error, as of its ConfusionMatrixMeasure unit, hence, it performs an inverse operation in case the given unit is maximal with a highly-fit model.
*/

class Evaluation_API ConfusionMatrixAnalytics: public AbstractAnalytics
{

public:

	/*!
	\brief Constructor.
	\param [in] aDataPackage the data package to be used in order to evaluate a model.
	\param [in] aUnit the unit of the evaluation.
	*/
	ConfusionMatrixAnalytics( muw::DataPackage* aDataPackage, QString aUnit );

	/*!
	\brief Destructor.
	*/
	~ConfusionMatrixAnalytics();

	/*!
	\brief Evaluates the given model by the predefined unit type.
	\param [in] aModel the model to evaluate.
	*/
	double evaluate( muw::AbstractModel* aModel ) override;

	/*!
	\return The ROC distance of the model's (1-sensitivity,specificity) from the ideal ROC value (top-left corner of the ROC curve).
	*/
	double rocDistance();

	/*!
	\return the F-score of the model based on input beta.
	\param [in] aBeta the beta value of the F-score calculation.
	*/
	double fScore( double aBeta = 2.0 );

	/*!
	\return The accuracy of the model.
	*/
	double acc();

	/*!
	\return The sensitivity of the model.
	*/
	double sns();

	/*!
	\return The specificity of the model.
	*/
	double spc();

	/*!
	\return The positive predictive value of the model.
	*/
	double ppv();

	/*!
	\return The negative predictive value of the model.
	*/
	double npv();

	/*!
	\return The matthew's correlation coefficient of the model.
	*/
	double mcc();

	/*!
	\return The ballanced accuracy of the model.
	*/
	double bacc();

	/*!
	\brief Returns all the implemented unit measures of ConfusionMatrixAnalytics of the model.
	*/
	QMap< QString, double > allValues() override;


private:

	/*!
	\brief Default constructor cannot be used.
	*/
	ConfusionMatrixAnalytics();

	/*!
	\brief Calculates true positive, true negative, false positive and false negative values for the moded, depending on its classifitcation type (binary or multi-class).
	*/
	void calculateAtomicErrors();

	/*!
	\brief Resets the confusion matrix.
	*/
	void resetConfusionMatrix();

	/*!
	\brief Resets the intermediate containers that store true positive, true negative, false positive, false negative values.
	*/
	void resetContainers();

private:

	muw::Array2D< unsigned int >*  mConfusionMatrix;         //!< The confusion matrix storing either a binary or a multi-class 2D array.
	ConfusionMatrixMeasure         mConfusionMatrixMeasure;  //!< The enum to store the unit of the confusion matrix measure.
	QVector< double >              mTPs;                     //!< The true positives.
	QVector< double >              mTNs;                     //!< The true negatives.
	QVector< double >              mFPs;                     //!< The false positives.
	QVector< double >              mFNs;                     //!< The false negatives
	double                         mDiagonalSum;             //!< The sum of the diagonal entries in the confusion matrix.
	double                         mSum;                     //!< The sum of all entries in the confusion matrix.
	bool                           mIsValid;                 //!< Stores if the confusion matrix has a valid state.
};

//-----------------------------------------------------------------------------

}
