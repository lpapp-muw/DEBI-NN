/*!
* \file
* This file is part of the DataRepresentation module.
*
* \remarks 
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/
#pragma once

#include <DataRepresentation/Export.h>
#include <DataRepresentation/Types.h>

namespace muw
{

//-----------------------------------------------------------------------------

template < typename Type >

/*!
* \brief The Array2D class is responsible to characterize a 2D array containing numerical types such as unsigned int, float and double.
*
* \details
*/
class DataRepresentation_API Array2D
{

public:

	/*!
	\brief Constructor.
	\param [in] aRowCount the row count of the array.
	\param [in] aColumnCount the column count of the array.
	*/
	Array2D( unsigned int aRowCount, unsigned int aColumnCount );

	/*!
	\brief Destructor.
	*/
	virtual ~Array2D();

	/*!
	\brief const operator ().
	\param [in] aX the row coordinate of the value to return with.
	\param [in] aY the column coordinate of the value to return with.
	\return the value of the array at aX,aY coordinates.
	*/
	const Type& operator () ( unsigned int aX, unsigned int aY ) const;

	/*!
	\brief operator ().
	\param [in] aX the row coordinate of the value to return with.
	\param [in] aY the column coordinate of the value to return with.
	\return the value of the array at aX,aY coordinates.
	*/
	Type& operator () ( unsigned int aX, unsigned int aY );

	/*!
	\brief increments the value of the array with 1 at coordinates aX,aY.
	\param [in] aX the row coordinate of the value to increment.
	\param [in] aY the column coordinate of the value to increment.
	*/
	void addEntry( unsigned int aX, unsigned int aY ) { operator() ( aX, aY ) += 1; }

	/*!
	\return the number of rows of the array.
	*/
	const unsigned int rowCount() const { return mRowCount; }

	/*!
	\return the number of columns of the array.
	*/
	const unsigned int columnCount() const { return mColumnCount; }

	/*!
	\brief same as const operator().
	\param [in] aX the row coordinate of the value to return with.
	\param [in] aY the column coordinate of the value to return with.
	\return the value of the array at aX,aY coordinates.
	*/
	const Type& at( unsigned int aX, unsigned int aY ) const { return operator()( aX, aY ); }

private:

	/*!
	\brief Default contructor not allowed.
	*/
	Array2D( const Array2D< Type >& );

	/*!
	\brief operator = not allowed.
	*/
	Array2D& operator = ( const Array2D< Type >& );

private:

	unsigned int mRowCount;     //!< The number of rows in the array.
	unsigned int mColumnCount;  //!< The number of columns in the array.
	Type* mArray;               //!< The array container.
};

//-----------------------------------------------------------------------------

}
