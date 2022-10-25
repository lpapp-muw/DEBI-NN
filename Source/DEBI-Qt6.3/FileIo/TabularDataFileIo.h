/*!
* \file This file is part of FileIo module.
*
* \remarks
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#pragma once

#include <FileIo/Export.h>
#include <DataRepresentation/TabularData.h>

namespace muw
{

//-----------------------------------------------------------------------------

/*!
* \brief The TabularDataFileIo class is responsible to load and save tabular data from and to CSV file format.
*
* \details Note: The current implementaiton assumes that the comma separator is ";".
*/

class FileIo_API TabularDataFileIo
{

public:

	/*!
	* \brief Default constructor.
	*/
	TabularDataFileIo();

	/*!
	* \brief Constructor.
	* \param [in] aWorkingDirectory The working directory of CSV files for loading/saving.
	*/
	TabularDataFileIo( QString aWorkingDirectory );

	/*!
	* \brief Destructor.
	*/
	~TabularDataFileIo();

	/*!
	* \brief Loads a CSV file to an input tabular data. Note that in case a working directory was not provided in the contructor, the path to the CSV file has to be absolute.
	* \param [in] aFileName The name of the CSV file (in case working directory was provided) or the path of the CSV file (in case default constructor was called) to load from.
	* \param [in] aTabularData the tabualr data to load to.
	*/
	void load( QString aFileName, muw::TabularData& aTabularData );

	/*!
	* \brief Saves an input tabular data into a CSV file. Note that in case a working directory was not provided in the contructor, the path to the CSV file has to be absolute.
	* \param [in] aFileName The name of the CSV file (in case working directory was provided) or the path of the CSV file (in case default constructor was called) to save to.
	* \param [in] aTabularData the tabualr data to ave from.
	*/
	void save( QString aFileName, muw::TabularData& aTabularData );

private:

	QString  mWorkingDirectory;  //!< The working directory of the file tabular data file IO.
	char     mCommaSeparator;    //!< Comma separator character for CSV file handling.

};

//-----------------------------------------------------------------------------

}
