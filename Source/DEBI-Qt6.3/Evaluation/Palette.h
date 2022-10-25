/*!
* \file
* This file is part of the Evaluation module.
*
* \remarks
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Boglarka Ecsedi, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#pragma once

#include <Evaluation/Export.h>
#include <QMap>
#include <QVector>
#include <QColor>
#include <cfloat>

namespace muw
{

//-----------------------------------------------------------------------------

/*!
* \brief The Palette class implements a simple color scale for rendering DEBI neural network components.
*
* \details
*/

class Evaluation_API Palette
{

public:

	/*!
	\brief Constructor.
	*/
	Palette() : mPalette(), mMin( DBL_MAX ), mMax( -DBL_MAX ) {}

	/*!
	\brief Destructor.
	*/
	~Palette() {}

	/*!
	\brief Sets the position-color control points and the required resolution for creating a palette.
	\param [in] aControlPoints holds position-color pairs, where a position shall be between 0.0 - 1.0, denoting a control point position on the scale of 0.0 - 1.0 of the palette. The palette is generated with a target resolution by interpolating colors in-between consecutive control points via linear interpolation.
	\param [in] aResolution the resolution to which the palette shall be generated.
	*/
	void setPaletteSettings( QMap< double, QColor > aControlPoints, int aResolution );

	/*!
	\brief Sets the min-max range of stretch the palette in-between.
	\param [in] aMin the minimum value to stretch the palette minimum to.
	\param [in] aMax the maximum value to stretch the palette maximum to.
	*/
	void setMinMax( double aMin, double aMax ) { mMin = aMin; mMax = aMax; }

	/*!
	\return Returns with a color corresponding the input value in the palette.
	\param [in] aValue the value which has to be mapped to a color in the palette.

	*/
	QColor paletteColor( double aValue );

private:

	QVector< QColor >  mPalette;  //!< The colors of the palette with a requested resolution.
	double             mMin;      //!< The minimum value to stretch the palette minimum to.
	double             mMax;      //!< The maximum value to stretch the palette maximum to.
	
};

//-----------------------------------------------------------------------------

}
