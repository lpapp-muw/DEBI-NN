/*! \mainpage Distance Encoding Biomorphic-Informational Neural Networks (DEBI-NN)
 *
 * \section intro_sec Introduction
 *
 * This Doxygen documentation describes the DEBI-NN architecture as implemented in C++.
 * Implementation was performed by researchers of the Medical University of Vienna (MUV), Vienna, Austria. The development is conducted at the Center for Medical Physics and Biomedical Engineering (main developer: Laszlo Papp, PhD, e-mail: laszlo.papp@meduniwien.ac.at), under the umbrella of projects, conducted by the Applied Quantum Computing (AQC) group (https://mpbmt.meduniwien.ac.at/en/research/quantum-computing/).
 * Note that the DEBI-NN describes a fully-connected neural network scheme. While we understand that - especially in the shadow of modern deep learning approaches - this model scheme may not be practically relevant for real-life applications, we consider that the DEBI-NN concept is worthy to investigate in scientific experimental settings.
 *
 *  \section intended_use_sec Intended Use
 * 
 * We explicitely state that the DEBI-NN source code does not describe any product and it is not intended to be used in any real-life, especially not in clinical settings, as it is purely a research tool for experimenting with biomorphic and informational neural network schemes and shall not be used partly of fully for rendering clinical decisions.
 * Please note that our paper describing DEBI-NNs as well as its predictive performance comparison with conventional NNs is currently in review. We believe in and represent high values of scientific integrity. Therefore, as long as our paper is in review and not accepted for publication, we do not recommend to rely on our code, and hence, we take no responsibility for any damage caused by the code.
 * With that being said, we strongly recommend to investigate the contents of this document and the source code only if you are the reviewer of our paper. Note that we will update this document with up-to-date information in case our paper undergoes a proper peer review and its publication stage changes.
 * 
 * 
 */


#include <Evaluation/DEBINNFactory.h>
#include "DEBINNCrossValidator.h"
#include "DEBINNRenderController.h"
#include <QtWidgets/QApplication>
#include <QSettings>
#include <QDir>

//-----------------------------------------------------------------------------

/*!
\brief Builds and cross-validates DEBI models as of the settings and dataset of the project folder.
\param [in] aProjectFolder the path to the project folder in which /Settings and /Dataset subfolders are located. The /Dataset/XXX subfolder must contain already generated and preprocessed training-validation folds where XXX denotes the given cohort fodler name.
*/
void buildAndCrossValidateDEBI( QString aProjectFolder )
{

	QDir dir ( aProjectFolder + "/Executions/" );
	auto subFolders = dir.entryList( QDir::Filter::NoDotAndDotDot | QDir::Filter::Dirs );

	for ( auto subFolder : subFolders )
	{
		QString executionFolderPath = aProjectFolder + "/Executions/" + subFolder + "/";
		//QString factorySettingFolderPath = aProjectFolder + "/" + subFolder + "/Settings/";

		qDebug() << "Processing execution" << executionFolderPath;
		muw::DEBINNCrossValidator DEBINNV( executionFolderPath );
		DEBINNV.execute();
	}
}

//-----------------------------------------------------------------------------

/*!
\brief Visualizes DEBI models located in the /Models subfolder of the project folder as of the settings in /Settings subfolder.
\param [in] aProjectFolder the path to the project folder in which /Settings and /Models subfolders are located.
*/
void visualizeModels( QString aProjectFolder )
{
	QSettings settings( aProjectFolder + "Models/Settings/settings.ini", QSettings::IniFormat );

	if ( !QFile::exists( settings.fileName() ) )
	{
		qWarning() << "File" << settings.fileName() << "doesn't exist!";
		exit( EXIT_FAILURE );
	}

	QVariantMap settingMap;
	for ( auto key : settings.allKeys() )
	{
		settingMap.insert( key, settings.value( key ) );
	}

	QVector< muw::DEBINN* > DEBIs;

	QStringList filters = { "*Model-*" };
	QStringList models = QDir( aProjectFolder + "/Models/" ).entryList( filters );

	for ( auto model : models )
	{
		muw::DEBINN DEBI( settingMap );
		DEBI.load( aProjectFolder + "/Models/" + model );
		DEBIs.push_back( new muw::DEBINN( DEBI ) );
	}

	muw::DEBINNRenderController* DEBINNRCT = new muw::DEBINNRenderController( settingMap );
	DEBINNRCT->setProjectFolder( aProjectFolder );
	DEBINNRCT->show();
	DEBINNRCT->createScenes( DEBIs );
}

//-----------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
	QApplication app( argc, argv );

	QString projectFolder;     // The project folder containing Settings, Dataset, Models and Screenshots subfolders.

	auto args = QCoreApplication::arguments();
	if ( args.size() > 1 )
		projectFolder = args[ 1 ];
	else
	{
		projectFolder = "c:/temp/DEBINN3.0/_forGIT/Examples/";  // Add your default project folder here.
		qDebug() << "Project folder not provided as argument. Going with default project folder" << projectFolder;
	}

	buildAndCrossValidateDEBI( projectFolder );    // Builds and cross-validates DEBI models based on the settings and dataset of projectFolder.

	//visualizeModels( projectFolder );              // Visualizes DEBI models located in /Models/ of projectFolder.

	app.setApplicationDisplayName( "DEBI - " + projectFolder );

	return app.exec();
}

//-----------------------------------------------------------------------------
