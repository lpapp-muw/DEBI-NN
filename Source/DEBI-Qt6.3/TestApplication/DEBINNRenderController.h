/*!
* \file
* This file is part of the TestApplication module.
*
* \remarks
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#pragma once

#include <QMainWindow>
#include <Evaluation/DEBINN.h>
#include <Evaluation/DEBINNRenderer.h>
#include <Qt3DExtras/qorbitcameracontroller.h>
#include <Qt3DRender/QRenderCapture>
#include <Qt3DRender/QLayer>
#include <Qt3DRender/QLayerFilter>
#include <QSettings>
#include <QString>
#include <QEventLoop>
#include <functional>

namespace Ui {
class DEBINNRenderController;
}

namespace muw
{

//-----------------------------------------------------------------------------

/*!
* \brief The DEBINNRenderController class is responsible to provide a GUI to visualize trained DEBINN models relying on QT3D rendering mechanisms.
*
* \details DEBINNRenderController class Allows users to interact with DEBINN models (e.g., rotate) in 3D as well as to observe multiple trained DEBINN models in case they were saved out during the training process.
* The class is also able to capture 3D views and save them to PNG format. It is intended that this class is further implemented to support the interpretation and debugging process of DENI models and training processes. As such, visualizing layer boundaries,
* selecting individual objects (e.g., neurons, lines, layers), visualizing input sample evaluation processes via their respective action potentials
*/

class DEBINNRenderController : public QMainWindow
{

	Q_OBJECT

public:


	/*!
	\brief Constructor.
	\param [in] aSettings the settings for rendering DEBINN models (e.g., parameters for lines, shperes).
	\param [in] aParent the parent window of the GUI application.
	*/
	DEBINNRenderController( QVariantMap aSettings, QWidget* aParent = nullptr );
	virtual ~DEBINNRenderController();

	/*!
	\brief Sets the settings container to render DEBINN models.
	\param [in] aSettings the settings for rendering DEBINN models.
	*/
	void setSettings( QVariantMap aSettings );

	/*!
	\brief Creates scenes for each input DEBI model to visualize.
	\param [in] aDEBIs the DEBI models loaded up for visualization.
	*/
	void createScenes( QVector< muw::DEBINN* > aDEBIs );

	/*!
	\brief Sets the project folder for the render controller.
	\param [in] aProjectFolder the project folder of the render controller to load/save content.
	*/
	void setProjectFolder( QString aProjectFolder ) { mProjectFolder = aProjectFolder; mView->setTitle( "DEBI - " + mProjectFolder ); this->setWindowTitle( "DEBI Render controller - " + mProjectFolder ); };

public slots:

	/*!
	\brief Starts a rotational animation of the currently visualized DEBINN model.
	*/
	void startAnimation();

	/*!
	\brief Sops an ongoing animation.
	*/
	void stopAnimation();

	/*!
	\brief Resets an ongoing animation to its default view state.
	*/
	void resetAnimation();

	/*!
	\brief Sets the linear speed parameter for the animation.
	\param [in] aLinearSpeed the linear speed of the animation.
	*/
	void setLinearSpeed( double aLinearSpeed );

	/*!
	\brief Sets the look speed parameter for the animation.
	\param [in] aLookSpeed the linar speed of the animation.
	*/
	void setLookSpeed( double aLookSpeed );

	/*!
	\brief Sets the model intex to the input one to visualize the respective DEBINN model.
	\param [in] aModelIndex the index of the DEBINN model to visualize.
	*/
	void selectModel( int aModelIndex );

	/*!
	\brief Saves the screen content.
	\param [in] aViewIndex the view index corresponding to the active model to visualize before grabbing.
	\param [in] aOnSaved function to grab the screen.
	*/
	void grabScreen( int aViewIndex = -1, std::function< void() > aOnSaved = nullptr );

	/*!
	\brief Scrolls through the DEBINN models automatically and grabs their screen to ave them to PNG format.
	*/
	void autoPilotModelScroll();


private:
	Ui::DEBINNRenderController*          mUI;						//!< The QT UI form.
	QVariantMap                          mSettings;					//!< Settings for rendering parameters.
	Qt3DExtras::Qt3DWindow*              mView;						//!< The 3D view. 
	Qt3DExtras::QOrbitCameraController*  mCamController;			//!< The camera controller the user cna interact with.
	bool                                 mIsStop;					//!< Holds the value of whether an ongoing animation shall stop.
	QVector< Qt3DCore::QEntity* >        mScenes;					//!< Buffer container for multiple DEBINN model renders.
	Qt3DCore::QEntity                    mRootEntity;				//!< The root entity to render.
	int                                  mModelIndex;				//!< The current model index to visualize.
	QString                              mProjectFolder;			//!< The project folder in which the renderer loads/saves content.
	int                                  mGlobalScreenGrabCounter;  //!< Counter to increment during saving out screenshots in one execution.
	Qt3DRender::QRenderCapture*          mRenderCapture;			//!< The screen capture of the 3D view.
};

//-----------------------------------------------------------------------------

}