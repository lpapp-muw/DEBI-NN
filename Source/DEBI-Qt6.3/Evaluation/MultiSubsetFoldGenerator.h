/* !
*\file
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
#include <Evaluation/DataPackage.h>
#include <QMap>
#include <QString>
#include <random>

namespace muw
{

//-----------------------------------------------------------------------------

/*!
* \briefThe  MultiSubsetFoldGenerator class is responsible to perform data split operations to support e.g. train-validate cross-validation processes.
*
* \details The implementation is able to split an input DataPackage to multiple numbers of subsets by their expected ratios.
* Note that a given subset ratio will be reflected on the minority subgroup size. Therefore, a particular ratio (0.0 < r ; 1.0) will result in balanced subsets as of the minority subgroup size. In case an imbalanced split is desired, assign ratio -1 to the given subset.
* Example: Consider that the subset defition for splitting is: "Balanced",0.2 and "Imbalanced",-1. This will result in a subset having 0.2 * the minority subgroup size number of samples from all label subgroups as well as a remnant, but imbalanced subset.
*/

class Evaluation_API MultiSubsetFoldGenerator
{

public:

	/*!
	* \brief Constructor.
	* \param [in] aDataPackage the data package to split into subsets.
	* \param [in] aSubsetDefinitions a map containing QString-double pairs describing a subset name and ratio for splitting the input data package.
	* \param [in] aFoldCount the number of folds the generator shall randomly split the input data package.
	*/
	MultiSubsetFoldGenerator( muw::DataPackage* aDataPackage, QMap< QString, double > aSubsetDefinitions, int aFoldCount, unsigned int aRandomSeed );
	
	/*!
	* \brief Constructor.
	* \param [in] aDataPackage the data package to split into subsets.
	* \param [in] aSubsetDefinitions a map containing QString-double pairs describing a subset name and ratio for splitting the input data package.
	* \param [in] aFoldConfigurationFilePath a CSV file containing pregenerated fold configurations.
	*/
	MultiSubsetFoldGenerator( muw::DataPackage* aDataPackage, QMap< QString, double > aSubsetDefinitions, QString aFoldConfigurationFilePath );

	/*!
	* \brief Destructor.
	*/
	~MultiSubsetFoldGenerator() { delete mGenerator; mGenerator = nullptr; }

	/*!
	* \brief Performs the random subset configuration generation. Note that this step does not generate actual data packages, only generates configurations for data package subsets.
	*/
	void execute();

	/*!
	* \brief Generates data package subsets as of the subset defitions for the givel fold.
	* \param [in] aFold the fold index to generate.
	* \return the data packages corresponding to the subset definitions for the given fold.
	*/
	QMap< QString, muw::DataPackage* > fold( int aFold );

	/*!
	* \brief Returns the generated fold splitting configurations.
	* \return the generated fold splitting configuraiton in tabular data form.
	*/
	muw::TabularData& foldDefinitions() { return mFoldDefinitions; }

	/*!
	* \return the number of folds generated.
	*/
	int foldCount() { return mFoldCount; }


private:

	muw::DataPackage*        mDataPackage;				 //!< The input data package to split to multiple folds.
	QMap< QString, double >  mSubsetDefinitions;		 //!< The container to define which data subset shall have which split ratio.
	int                      mFoldCount;				 //!< The number of folds to generate.
	QMap< QString, double >  mAlignedSubsetDefinitions;  //!< The subset ratios aligned to the minority subgroup of the input dataset.
	QMap< QString, int >     mAlignedSubsetCounts;		 //!< The subset counts aligned to the minority subgroup of the input dataset.
	muw::TabularData         mFoldDefinitions;			 //!< The random split configuraitons for all folds.
	std::mt19937*            mGenerator;                 //!< Random generator.
	bool                     mIsValid;					 //!< True if the subset generator managed to generate the desired amount of subsets from the input dataset.
	
};

}