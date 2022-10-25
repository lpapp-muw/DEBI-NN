/*!
* \file
* This file is part of TestApplication module.
* Member function definitions for DEBINNCRenderController class.
*
* \remarks
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#include "DEBINNRenderController.h"
#include <ui_DEBINNRenderController.h>
#include <Qt3DRender/qpointlight.h>
#include <Qt3DAnimation>
#include <QScreen>

namespace muw
{

//-----------------------------------------------------------------------------

DEBINNRenderController::DEBINNRenderController( QVariantMap aSettings, QWidget* aParent )
:
	QMainWindow( aParent ),
	mUI( new Ui::DEBINNRenderController ),
	mSettings( aSettings ),
	mView( nullptr ),
	mCamController( nullptr ),
	mIsStop( true ),
	mScenes(),
	mModelIndex( 0 ),
	mProjectFolder(),
	mGlobalScreenGrabCounter( 0 ),
	mRenderCapture( nullptr )
{
	mUI->setupUi( this );

	mView = new Qt3DExtras::Qt3DWindow();
	auto backgroundColor = mSettings.value( "Renderer/BackgroundColor" ).toList();

	mView->setRootEntity( &mRootEntity );

	mView->defaultFrameGraph()->setBuffersToClear( Qt3DRender::QClearBuffers::AllBuffers );
	mView->defaultFrameGraph()->setClearColor( QColor( backgroundColor.at( 0 ).toInt(), backgroundColor.at( 1 ).toInt(), backgroundColor.at( 2 ).toInt() ) );
	Qt3DRender::QCamera* camera = mView->camera();

	mRenderCapture = new Qt3DRender::QRenderCapture;

	mView->activeFrameGraph()->setParent( mRenderCapture );
	mView->setActiveFrameGraph( mRenderCapture );

	int orthoMargin = 35.0f;
	camera->lens()->setOrthographicProjection( -orthoMargin, orthoMargin, -orthoMargin, orthoMargin, -10000, 10000.0f );

	resetAnimation();

	mCamController = new Qt3DExtras::QOrbitCameraController( &mRootEntity );
	mCamController->setCamera( camera );
	mCamController->setLinearSpeed( mUI->doubleSpinBoxLinearSpeed->value() );
	mCamController->setLookSpeed( mUI->doubleSpinBoxLookSpeed->value() );

	Qt3DCore::QEntity* lightEntity1 = new Qt3DCore::QEntity( mView->camera() );
	Qt3DRender::QPointLight* light1 = new Qt3DRender::QPointLight( lightEntity1 );
	light1->setColor( "white" );
	light1->setIntensity( 1.0 );

	lightEntity1->addComponent( light1 );
	Qt3DCore::QTransform* lightTransform1 = new Qt3DCore::QTransform( lightEntity1 );
	lightTransform1->setTranslation( QVector3D( 0, 0, 0 ) );
	lightEntity1->addComponent( lightTransform1 );

	mView->setWidth( 1000 );
	mView->setHeight( 1000 );
	mView->show();

	connect( mUI->pushButtonStart, SIGNAL( released() ), this, SLOT( startAnimation() ) );
	connect( mUI->pushButtonStop, SIGNAL( released() ), this, SLOT( stopAnimation() ) );
	connect( mUI->pushButtonReset, SIGNAL( released() ), this, SLOT( resetAnimation() ) );
	connect( mUI->scrollbarModels, SIGNAL( valueChanged( int ) ), this, SLOT( selectModel( int ) ) );
	connect( mUI->doubleSpinBoxLinearSpeed, SIGNAL( valueChanged( double ) ), this, SLOT( setLinearSpeed( double ) ) );
	connect( mUI->doubleSpinBoxLookSpeed, SIGNAL( valueChanged( double ) ), this, SLOT( setLookSpeed( double ) ) );
	connect( mUI->pushButtonSaveView, SIGNAL( released() ), this, SLOT( grabScreen() ) );
	connect( mUI->pushButtonSaveViewBatch, SIGNAL( released() ), this, SLOT( autoPilotModelScroll() ) );
}

//-----------------------------------------------------------------------------

DEBINNRenderController::~DEBINNRenderController()
{
	delete mUI;
	delete mView;
	delete mRenderCapture;
	for (auto scene : mScenes)
	{
		delete scene;
	}
}

//-----------------------------------------------------------------------------

void DEBINNRenderController::createScenes( QVector< muw::DEBINN* > aDEBIs )
{
	mUI->scrollbarModels->setDisabled( false );

	auto* matRoot = muw::DEBINNRenderer::materialsRootEntity();
	matRoot->setParent( &mRootEntity );

	int m = 0;
	for (auto snn : aDEBIs)
	{
		++m;
		muw::DEBINNRenderer mSNNR( mSettings );

		auto scene = mSNNR.createScene( snn );

		mScenes.push_back( scene );
		qDebug() << "New scene created" << m << "/" << aDEBIs.size();
	}

	mScenes[ mModelIndex ]->setParent( &mRootEntity );

	mUI->scrollbarModels->setMinimum( 0 );
	mUI->scrollbarModels->setMaximum( mScenes.size() - 1 );

	if (mScenes.size() == 1) mUI->scrollbarModels->setDisabled( true );
}


//-----------------------------------------------------------------------------

void DEBINNRenderController::setSettings( QVariantMap aSettings )
{
	mSettings = aSettings;
}

//-----------------------------------------------------------------------------

void DEBINNRenderController::startAnimation()
{
	mIsStop = false;

	mView->camera()->setUpVector( QVector3D( 0, 1, 0 ) );

	QVector3D vector3D = mView->camera()->position();
	qreal r = qSqrt( vector3D.x() * vector3D.x() + vector3D.z() * vector3D.z() );
	qreal dZ = qAsin( vector3D.z() / r );
	qreal dX = qAcos( vector3D.x() / r );
	qreal delta = mUI->doubleSpinBoxDelta->value();

	while (!mIsStop)
	{
		dX += delta;
		dZ += delta;
		qreal z = ( r * qSin( dZ ) );
		qreal x = ( r * qCos( dX ) );

		vector3D.setX( x );
		vector3D.setZ( z );

		mView->camera()->setPosition( vector3D );
		mView->camera()->setUpVector( QVector3D( 0, 1, 0 ) );


		QEventLoop eventloop;
		QTimer::singleShot( 15, &eventloop, SLOT( quit() ) );
		eventloop.exec();
	}
}

//-----------------------------------------------------------------------------

void DEBINNRenderController::stopAnimation()
{
	mIsStop = true;
}

//-----------------------------------------------------------------------------

void DEBINNRenderController::resetAnimation()
{
	mIsStop = true;
	mView->camera()->setViewCenter( QVector3D( 0, 0, 0 ) );
	mView->camera()->setPosition( QVector3D( -359.0f, 0.0f, 0.0f ) );
	mView->camera()->setUpVector( QVector3D( 0, 1, 0 ) );
}

//-----------------------------------------------------------------------------

void DEBINNRenderController::setLinearSpeed( double aLinearSpeed )
{
	mCamController->setLinearSpeed( aLinearSpeed );
}

//-----------------------------------------------------------------------------

void DEBINNRenderController::setLookSpeed( double aLookSpeed )
{
	mCamController->setLookSpeed( aLookSpeed );
}

//-----------------------------------------------------------------------------

void DEBINNRenderController::selectModel( int aModelIndex )
{
	QApplication::setOverrideCursor( Qt::WaitCursor );

	mScenes[ mModelIndex ]->setParent( static_cast< Qt3DCore::QNode* >( nullptr ) );

	mModelIndex = aModelIndex;

	mScenes[ mModelIndex ]->setParent( &mRootEntity );

	QApplication::restoreOverrideCursor();
}

//-----------------------------------------------------------------------------

void DEBINNRenderController::grabScreen( int aViewIndex, std::function< void() > aOnSaved )
{
	QDir dir;
	if (!dir.exists( mProjectFolder + "/Screenshots/" )) dir.mkpath( mProjectFolder + "/Screenshots/" );

	mView->renderSettings()->setRenderPolicy( Qt3DRender::QRenderSettings::RenderPolicy::Always );

	Qt3DRender::QRenderCaptureReply* captureReply = mRenderCapture->requestCapture();
	connect( captureReply, &Qt3DRender::QRenderCaptureReply::completed, mRenderCapture, [=]
	{
		mView->renderSettings()->setRenderPolicy( Qt3DRender::QRenderSettings::RenderPolicy::OnDemand );

		QImage image = captureReply->image();
		captureReply->deleteLater();

		image.save( mProjectFolder + "/Screenshots/Screen-" + QString::number( mGlobalScreenGrabCounter ) + ".png", "PNG", 85 );

		if (aOnSaved)
			aOnSaved();
	} );

	++mGlobalScreenGrabCounter;
}

//-----------------------------------------------------------------------------


void DEBINNRenderController::autoPilotModelScroll()
{
	mView->camera()->setUpVector( QVector3D( 0, 1, 0 ) );

	struct RenderState
	{
		DEBINNRenderController* controller;
		qreal r, dZ, dX, deltaMultiplier, delta;
		QVector3D vector3D;
		int modelIndex = 0;

		RenderState( DEBINNRenderController* aController )
		{
			controller = aController;

			vector3D = controller->mView->camera()->position();
			r = qSqrt( vector3D.x() * vector3D.x() + vector3D.z() * vector3D.z() );
			dZ = qAsin( vector3D.z() / r );
			dX = qAcos( vector3D.x() / r );
			deltaMultiplier = 1.0;
			delta = controller->mUI->doubleSpinBoxDelta->value();
		}

		void captureNext()
		{
			qreal currentDelta = delta * deltaMultiplier;

			qreal myX = dX + modelIndex * currentDelta;
			qreal myZ = dZ + modelIndex * currentDelta;
			qreal z = ( r * qSin( myZ ) );
			qreal x = ( r * qCos( myX ) );

			auto myVector3D = vector3D;
			myVector3D.setX( x );
			myVector3D.setZ( z );

			controller->mView->camera()->setPosition( myVector3D );
			controller->mView->camera()->setUpVector( QVector3D( 0, 1, 0 ) );

			controller->selectModel( modelIndex );

			QTimer::singleShot( 100, controller, [=]
			{

				controller->grabScreen( -1, [=]
				{
					modelIndex++;

					if (modelIndex < controller->mScenes.size())
					{
						captureNext();
					}
					else
					{
						controller->selectModel( controller->mUI->scrollbarModels->value() );
						QApplication::restoreOverrideCursor();
						controller->setEnabled( true );

						delete this;
					}
				} );
			} );
		}
	};

	setEnabled( false );
	QApplication::setOverrideCursor( Qt::WaitCursor );

	RenderState* state = new RenderState( this );
	state->captureNext();

}

//-----------------------------------------------------------------------------

}
