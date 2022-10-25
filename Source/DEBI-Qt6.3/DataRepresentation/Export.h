/*!
 * \file
 * Storage-class information setting for DataRepresentation module.
 *
 * \remarks
 *
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#pragma once

#define DataRepresentation_API

#if 0
#if defined( EXPORT_DataRepresentation )
#	define DataRepresentation_API __declspec(dllexport)
#else
#	define DataRepresentation_API __declspec(dllimport)
#endif
#endif
