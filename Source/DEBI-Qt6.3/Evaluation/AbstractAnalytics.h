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
#include <Evaluation/AbstractModel.h>
#include <Evaluation/DataPackage.h>
#include <QSettings>
#include <QString>

namespace muw
{

//-----------------------------------------------------------------------------

/*!
* \brief The AbstractAnalytics class is a base class for all other analytics classes to provide a unified interface for optimizers (algorithms that train models).
* AbstractAnalytics requires a DataPackage to use it for evaluating a model which is provided separately.
*
* \details
*/

class Evaluation_API AbstractAnalytics
{

public:

	/*!
	\brief Constructor.
	\param [in] aDataPackage the data package to utilize in order to evaluate a model with the given analytics.
	\param [in] aUnit the unit of analytics. A particular analytics family may have multiple units (e.g. accuracy or F2 score).
	*/
	AbstractAnalytics( muw::DataPackage* aDataPackage, QString aUnit = "" ): mDataPackage( aDataPackage ), mUnit( aUnit ) {}

	/*!
	\brief Evaluates the input model by the data package of the given analytics object.
	\return the fitness or error (analytics-specific) of the input model.
	*/
	virtual double evaluate( muw::AbstractModel* aModel ) = 0;

	/*!
	\brief Destructor.
	*/
	virtual ~AbstractAnalytics();

	/*!
	\brief Returns all values (units) of the given analytics type.
	\return the key-value map of all values the given analytics type can calculate for a model.
	*/
	virtual QMap< QString, double > allValues() = 0;


	/*!
	\return the unit string of the given analytics.
	*/
	const QString& unit() const
	{
		return mUnit;
	};

	/*!
	\return The stored data package to use for evaluating models.
	*/
	muw::DataPackage* dataPackage() { return mDataPackage; }

private:

	/*!
	\brief Default constructor is not allowed to be used.
	*/
	AbstractAnalytics();

protected:

	muw::DataPackage*  mDataPackage;  //!< The data package to use for evaluating a model.
	QString            mUnit;         //!< The unit of the given analytics to be used for evaluating the model.
};

//-----------------------------------------------------------------------------

}
