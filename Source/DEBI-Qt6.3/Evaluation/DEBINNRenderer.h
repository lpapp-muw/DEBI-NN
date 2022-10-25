/*!
* \file
* This file is part of the Evaluation module.
*
* \remarks
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
* Boglarka Ecsedi, Center for Medical Physics and Biomedical engineering, MUV, Austria.
*/

#pragma once

#include <Evaluation/Export.h>
#include <Evaluation/Palette.h>
#include <Qt3DCore/QEntity>
#include <Qt3DRender/QCamera>
#include <Qt3DRender/QCameraLens>
#include <Qt3DCore/QTransform>
#include <Qt3DCore/QAspectEngine>
#include <Qt3DInput/QInputAspect>
#include <Qt3DRender/QRenderAspect>
#include <Qt3DExtras/QForwardRenderer>
#include <Qt3DExtras/QPhongMaterial>
#include <Qt3DExtras/QCylinderMesh>
#include <Qt3DExtras/QSphereMesh>
#include <Qt3DRender/QPointLight>
#include <Qt3DRender/QRenderAspect>
#include <Qt3DExtras/QForwardRenderer>
#include <Qt3DExtras/QPhongAlphaMaterial>
#include <Qt3DExtras/QDiffuseSpecularMaterial>
#include <Qt3DExtras/QCylinderMesh>
#include <Qt3DExtras/QSphereMesh>
#include <QPropertyAnimation>
#include <Qt3DExtras/Qt3DWindow>
#include <Evaluation/DEBINN.h>

namespace muw
{

//-----------------------------------------------------------------------------

/*!
* \brief The DEBINNRenderer class implements a simple 3D rendering of a DEBI neural network.
*
* \details
*/

class Evaluation_API DEBINNRenderer
{

public:

	/*!
	\brief LineType enum to handle axon and dendrite lines. Axon lines are between a DEBI neuron's soma and axon. Dendrite lines are weights in between an axon and a next layer soma.
	*/
	enum class LineType
	{
		Axon = 0,
		Dendrite
	};

	/*!
	\brief Constructor.
	\param [in] aSettings the settings container storing parameters for the rendering process.
	*/
	DEBINNRenderer( QVariantMap aSettings );

	/*!
	\brief Destructor.
	*/
	~DEBINNRenderer();

	/*!
	\return The setting container of the renderer.
	*/
	const QVariantMap& settings() const { return mSettings; }

	/*!
	\brief Creates a scene based on the input DEBI neural network and the internal rendering settings.
	\param [in] aDEBINN the DEBI neural network to render.
	*/
	Qt3DCore::QEntity* createScene( muw::DEBINN* aDEBINN );

	static Qt3DCore::QEntity* materialsRootEntity() { return &mMaterialsRootEntity; }

private:

	/*!
	\brief Initializes the renderer by reading out parameters from its setting container.
	*/
	void initialize();

	/*!
	\brief Renders one DEBI neuron to the scene.
	\param [in] aDEBINeuron the DEBI neuron to render as of its spatial coordinates.
	*/
	void renderDebiNeuron( std::shared_ptr< muw::DEBINeuron > aDEBINeuron );

	/*!
	\brief Renders a line in the scene. Used to draw connections between somas and axons as well as to render a wireframe around the DEBI neural network.
	\param [in] aStart the start coordinate of the line.
	\param [in] aEnd the end coordinate of the line.
	\param [in] aColor the color properties of the line.
	*/
	void drawLine( const QVector3D& aStart, const QVector3D& aEnd, const QColor& aColor, Qt3DCore::QEntity* aRootEntity, LineType aLineType );

	/*!
	\brief Really add lines added via drawLine() into the scene. Called at the very end of scene creation.
	*/
	void instantiateLines();

	void instantiateLinesForMaterial( Qt3DExtras::QPhongAlphaMaterial* aMaterial, const QList< QVector3D >& aLines );

private:
	
	QVariantMap                                               mSettings;                    //!< The setting container of the renderer.
	Qt3DCore::QEntity*                                        mRootEntity;                  //!< The scene entity in which the drawing is performed.
	QMap< QString, Qt3DExtras::QDiffuseSpecularMaterial* >    mMaterials;                   //!< Material properties of the renderer.
	QMap< QString, QColor >                                   mColors;                      //!< Color properties of the renderer.
	Qt3DExtras::QDiffuseSpecularMaterial*                     mNeuronAxonLineMaterial;      //!< The material of the axon lines.
	Qt3DExtras::QDiffuseSpecularMaterial*                     mNeuronDendriteLineMaterial;  //!< The material of the dendrite lines.
	QVector< double >                                         mNNCenter;                    //!< The center of the DEBI neural network.
	Palette                                                   mWeightPalette;               //!< The palette to color weights in the DEBI neural network.
	Qt3DExtras::QSphereMesh*                                  mSphereMesh;                  //!< The mesh of all speheres rendered.
	QMap< QString, QList< QVector3D > >                       mLinesByMaterial;             //!< Line definitons by material type.
	static QMap< QString, Qt3DExtras::QPhongAlphaMaterial* >  mLineMaterials;               //!< Line materials.
	static Qt3DCore::QEntity mMaterialsRootEntity;                                          //!< The root entity for materials.
};

//-----------------------------------------------------------------------------

}