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
#include <QString>


namespace muw
{

//-----------------------------------------------------------------------------

/*!
* \brief The DEBIStemCell is a base class of the DEBINeuron class. Currently, it is acting as an architectural placeholder, since it is assumed that future DEBI neural network implementations may allow different neuron types to exist in them.
*
* \details
*/

class Evaluation_API DEBIStemCell
{

public:

	/*!
	\brief Constructor.
	\param [in] aCellType the type of the neural cell.
	*/
	DEBIStemCell( QString aCellType ) : mCellType( aCellType ) {}
	
	/*!
	\brief Destructor
	*/
	virtual ~DEBIStemCell() {}

	
private:

	/*!
	\brief Default constructor is not allowed.
	*/
	DEBIStemCell();

private:

	QString  mCellType;  //!< The type fo the neural cell in the DEBI neural network.

};

//-----------------------------------------------------------------------------

}
