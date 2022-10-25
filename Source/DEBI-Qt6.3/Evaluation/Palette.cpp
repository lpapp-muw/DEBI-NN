/*!
* \file
* Member class definitions of Palette. This file is part of the Evaluation module.
*
* \remarks
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Boglarka Ecsedi, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#include <Evaluation/Palette.h>

namespace muw
{

//-----------------------------------------------------------------------------

void Palette::setPaletteSettings(QMap< double, QColor > aControlPoints, int aResolution)
{
	mPalette.clear();

	for (int i = 1; i < aControlPoints.size(); ++i)
	{
		auto startControlPoint = aControlPoints.keys().at(i - 1);
		auto endControlPoint = aControlPoints.keys().at(i);

		int numberOfColors = double(aResolution) * (endControlPoint - startControlPoint);
		QColor startColor = aControlPoints.value(startControlPoint);
		QColor endColor = aControlPoints.value(endControlPoint);

		double stepRed = (((double(endColor.red()) - double(startColor.red())) / numberOfColors));
		double stepGreen = (((double(endColor.green()) - double(startColor.green())) / numberOfColors));
		double stepBlue = (((double(endColor.blue()) - double(startColor.blue())) / numberOfColors));
		double stepAlpha = (((double(endColor.alpha()) - double(startColor.alpha())) / numberOfColors));

		for (int j = 0; j < numberOfColors - 1; ++j)
		{
			int red = startColor.red() + (j * stepRed);
			int green = startColor.green() + (j * stepGreen);
			int blue = startColor.blue() + (j * stepBlue);
			int alpha = startColor.alpha() + (j * stepAlpha);

			mPalette.append(QColor(red, green, blue, alpha));
		}
	}
}

//-----------------------------------------------------------------------------

QColor Palette::paletteColor( double aValue )
{
	double normalizedWeight = (aValue - mMin) / (mMax - mMin);

	int paletteIndex = normalizedWeight * (mPalette.size() - 1);
	auto color = mPalette.at(paletteIndex);
	return color;

}

//-----------------------------------------------------------------------------

}