/*!
* \file
* Member function definitions of Array2D class. This file is part of DataRepresentation module.
*
* \remarks
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#include <DataRepresentation/Array2D.h>

namespace muw
{

//-----------------------------------------------------------------------------
template < typename Type >
Array2D< Type >::Array2D( unsigned int aRowCount, unsigned int aColumnCount )
:
	mRowCount( aRowCount ),
	mColumnCount( aColumnCount )
{
	if ( aRowCount > 0 && aColumnCount > 0 ) mArray = new Type[ aRowCount * aColumnCount ];
	for ( unsigned int arrayIndex = 0; arrayIndex < aRowCount * aColumnCount; ++arrayIndex )
	{
		mArray[ arrayIndex ] = 0;
	}
}

//-----------------------------------------------------------------------------
template < typename Type >
Array2D< Type >::~Array2D()
{
	delete[] mArray;
}

//-----------------------------------------------------------------------------
template < typename Type >
const Type& Array2D< Type >::operator() ( unsigned int aX, unsigned int aY ) const
{
	return mArray[ aY*mRowCount + aX ];
}

//-----------------------------------------------------------------------------
template < typename Type >
Type& Array2D< Type >::operator() ( unsigned int aX, unsigned int aY )
{
	return mArray[ aY*mRowCount + aX ];
}

//-----------------------------------------------------------------------------

// Template instantiations to export:
template class DataRepresentation_API Array2D< unsigned int >;
template class DataRepresentation_API Array2D< float >;
template class DataRepresentation_API Array2D< double >;

}