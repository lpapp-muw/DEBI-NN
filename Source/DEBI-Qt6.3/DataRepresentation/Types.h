/*!
 * \file
 * Integral types for DataRepresentation module.
 *
 * \remarks
 *
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#pragma once

#include <cstdint>

//-----------------------------------------------------------------------------

static bool isRandomSeed = false;

typedef unsigned int      uint;

// Integral types that match the word length.
#if defined( CC_MSVC )
typedef __int64           lint;
typedef unsigned __int64  ulint;
#else
typedef long int           lint;
typedef unsigned long int  ulint;
#endif
