/*!
* \file
* This file is part of the Evaluation module.
* Member class definitions of AbstractModel.
*
* \remarks
*
* \authors
* lpapp
*/

#include <Evaluation/AbstractModel.h>

namespace muw
{

//-----------------------------------------------------------------------------

AbstractModel::AbstractModel( QSettings* aSettings )
:
	mSettings( aSettings ),
	mFeatureNames()
{
}

//-----------------------------------------------------------------------------

AbstractModel::~AbstractModel()
{
}

//-----------------------------------------------------------------------------

}
