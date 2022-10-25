/*!
* \file
* Member class definitions of DEBINNRenderer. This file is part of the Evaluation module.
*
* \remarks
*
* \copyright Copyright 2022 Medical University of Vienna, Vienna, Austria. All rights reserved.
*
* \authors
* Laszlo Papp, PhD, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
* Boglarka Ecsedi, Center for Medical Physics and Biomedical Engineering, MUV, Austria.
*/

#include <Evaluation/DEBINNRenderer.h>
#include <Qt3DExtras/QCuboidMesh>
#include <Qt3DRender/QMaterial>
#include <Qt3DRender/QObjectPicker>
#include <Qt3DExtras/QDiffuseSpecularMaterial>
#include <Qt3DExtras/QCuboidMesh>
#include <Qt3DExtras/QCuboidGeometry>
#include <Qt3DRender/QLayer>

static const int ambientDarker = 220;

namespace muw
{

QMap< QString, Qt3DExtras::QPhongAlphaMaterial* > DEBINNRenderer::mLineMaterials;
Qt3DCore::QEntity DEBINNRenderer::mMaterialsRootEntity;

//-----------------------------------------------------------------------------

DEBINNRenderer::DEBINNRenderer( QVariantMap aSettings )
:
	mSettings(aSettings),
	mRootEntity(new Qt3DCore::QEntity),
	mMaterials(),
	mNeuronAxonLineMaterial(new Qt3DExtras::QDiffuseSpecularMaterial),
	mNeuronDendriteLineMaterial(new Qt3DExtras::QDiffuseSpecularMaterial),
	mWeightPalette()
{
	initialize(); 
}

//-----------------------------------------------------------------------------

DEBINNRenderer::~DEBINNRenderer()
{
}

//-----------------------------------------------------------------------------

void DEBINNRenderer::initialize()
{
	QString keyOfNeuralNetworkWeights = mSettings.value("Renderer/Roles/NeuralNetworkWeights").toString();
	int controlPointCount = mSettings.value("Renderer/" + keyOfNeuralNetworkWeights + "/ControlPointCount").toInt();
	QMap< double, QColor > controlPoints;
	for (int i = 0; i < controlPointCount; ++i)
	{
		QVariantList controlPointRow = mSettings.value("Renderer/" + keyOfNeuralNetworkWeights + "/C" + QString::number(i)).toList();
		double controlPointValue = controlPointRow.at(0).toDouble();
		QColor controlPointColor = QColor(controlPointRow.at(1).toInt(), controlPointRow.at(2).toInt(), controlPointRow.at(3).toInt(), controlPointRow.at(4).toInt());
		controlPoints.insert(controlPointValue, controlPointColor);
	}

	int resolution = mSettings.value("Renderer/" + keyOfNeuralNetworkWeights + "/Resolution").toInt();

	mWeightPalette.setPaletteSettings(controlPoints, resolution);

	mMaterials["Input/Soma"] = new Qt3DExtras::QDiffuseSpecularMaterial(mRootEntity);
	QVariantList inputSomaColor = mSettings.value("Renderer/Materials/Color/Input/Soma").toList();
	QColor color = QColor(inputSomaColor.at(0).toInt(), inputSomaColor.at(1).toInt(), inputSomaColor.at(2).toInt(), inputSomaColor.at(3).toInt());
	mMaterials["Input/Soma"]->setShareable( true );
	mMaterials["Input/Soma"]->setDiffuse(color);
	mMaterials["Input/Soma"]->setAmbient(color.darker(ambientDarker));

	mMaterials["Input/Axon"] = new Qt3DExtras::QDiffuseSpecularMaterial(mRootEntity);
	QVariantList inputAxonColor = mSettings.value("Renderer/Materials/Color/Input/Axon").toList();

	color = QColor(inputAxonColor.at(0).toInt(), inputAxonColor.at(1).toInt(), inputAxonColor.at(2).toInt(), inputAxonColor.at(3).toInt());
	mMaterials["Input/Axon"]->setShareable( true );
	mMaterials["Input/Axon"]->setDiffuse(color);
	mMaterials["Input/Axon"]->setAmbient(color.darker(ambientDarker));

	mMaterials["Hidden/Soma"] = new Qt3DExtras::QDiffuseSpecularMaterial(mRootEntity);
	QVariantList hiddenSomaColor = mSettings.value("Renderer/Materials/Color/Hidden/Soma").toList();

	color = QColor(hiddenSomaColor.at(0).toInt(), hiddenSomaColor.at(1).toInt(), hiddenSomaColor.at(2).toInt(), hiddenSomaColor.at(3).toInt());
	mMaterials["Hidden/Soma"]->setShareable( true );
	mMaterials["Hidden/Soma"]->setDiffuse(color);
	mMaterials["Hidden/Soma"]->setAmbient(color.darker(ambientDarker));

	mMaterials["Hidden/Axon"] = new Qt3DExtras::QDiffuseSpecularMaterial(mRootEntity);
	QVariantList hiddenAxonColor = mSettings.value("Renderer/Materials/Color/Hidden/Axon").toList();

	color = QColor(hiddenAxonColor.at(0).toInt(), hiddenAxonColor.at(1).toInt(), hiddenAxonColor.at(2).toInt(), hiddenAxonColor.at(3).toInt());
	mMaterials["Hidden/Axon"]->setShareable( true );
	mMaterials["Hidden/Axon"]->setDiffuse(color);
	mMaterials["Hidden/Axon"]->setAmbient(color.darker(ambientDarker));

	mMaterials["Output/Soma"] = new Qt3DExtras::QDiffuseSpecularMaterial(mRootEntity);
	QVariantList outputSomaColor = mSettings.value("Renderer/Materials/Color/Output/Soma").toList();

	color = QColor(outputSomaColor.at(0).toInt(), outputSomaColor.at(1).toInt(), outputSomaColor.at(2).toInt(), outputSomaColor.at(3).toInt());
	mMaterials["Output/Soma"]->setShareable( true );
	mMaterials["Output/Soma"]->setDiffuse(color);
	mMaterials["Output/Soma"]->setAmbient(color.darker(ambientDarker));

	mMaterials["Output/Axon"] = new Qt3DExtras::QDiffuseSpecularMaterial(mRootEntity);
	QVariantList outputAxonColor = mSettings.value("Renderer/Materials/Color/Output/Axon").toList();

	color = QColor(outputAxonColor.at(0).toInt(), outputAxonColor.at(1).toInt(), outputAxonColor.at(2).toInt(), outputAxonColor.at(3).toInt());
	mMaterials["Output/Axon"]->setShareable( true );
	mMaterials["Output/Axon"]->setDiffuse(color);
	mMaterials["Output/Axon"]->setAmbient(color.darker(ambientDarker));

	auto inputSomaAxonColor = mSettings.value( "Renderer/Materials/Color/Input/SomaAxon" ).toList();
	mColors[ "Input/SomaAxon" ] = QColor( inputSomaAxonColor.at( 0 ).toInt(), inputSomaAxonColor.at( 1 ).toInt(), inputSomaAxonColor.at( 2 ).toInt(), inputSomaAxonColor.at( 3 ).toInt() );

	auto hiddenSomaAxonColor = mSettings.value( "Renderer/Materials/Color/Hidden/SomaAxon" ).toList();
	mColors[ "Hidden/SomaAxon" ] = QColor( hiddenSomaAxonColor.at( 0 ).toInt(), hiddenSomaAxonColor.at( 1 ).toInt(), hiddenSomaAxonColor.at( 2 ).toInt(), hiddenSomaAxonColor.at( 3 ).toInt() );

	auto outputSomaAxonColor = mSettings.value( "Renderer/Materials/Color/Output/SomaAxon" ).toList();
	mColors[ "Output/SomaAxon" ] = QColor( outputSomaAxonColor.at( 0 ).toInt(), outputSomaAxonColor.at( 1 ).toInt(), outputSomaAxonColor.at( 2 ).toInt(), outputSomaAxonColor.at( 3 ).toInt() );

	mSphereMesh = new Qt3DExtras::QSphereMesh;
	mSphereMesh->setRadius( 1.0f );
	mSphereMesh->setGenerateTangents( true );
	mSphereMesh->setShareable( true );
}

//-----------------------------------------------------------------------------

Qt3DCore::QEntity* DEBINNRenderer::createScene(muw::DEBINN* aSNN)
{
	if (!aSNN->isValid())
	{
		aSNN->realign();
	}

	auto zRanges = aSNN->zRanges();

	mNNCenter = aSNN->center();

	auto DEBINeurons = aSNN->DEBINeurons();

	double globalMinWeight = aSNN->globalMinWeight();
	double globalMaxWeight = aSNN->globalMaxWeight();

	mWeightPalette.setMinMax( globalMinWeight, globalMaxWeight );

	auto layerCount = DEBINeurons.size();

	double neuronMinX = DBL_MAX;
	double neuronMinY = DBL_MAX;
	double neuronMinZ = DBL_MAX;

	double neuronMaxX = -DBL_MAX;
	double neuronMaxY = -DBL_MAX;
	double neuronMaxZ = -DBL_MAX;

	for (auto layer = 0; layer < layerCount; ++layer)
	{
		auto layerNeurons = DEBINeurons.at(layer);
		auto currentRange = zRanges.at(layer);

		for (int i = 0; i < layerNeurons.size(); ++i)
		{
			auto neuron = layerNeurons.at( i );
			renderDebiNeuron( neuron );

			auto soma = neuron->soma();
			auto axon = neuron->axon();

			if ( soma.x < neuronMinX ) neuronMinX = soma.x;
			if ( axon.x < neuronMinX ) neuronMinX = axon.x;
			if ( soma.y < neuronMinY ) neuronMinY = soma.y;
			if ( axon.y < neuronMinY ) neuronMinY = axon.y;
			if ( soma.z < neuronMinZ ) neuronMinZ = soma.z;
			if ( axon.z < neuronMinZ ) neuronMinZ = axon.z;

			if ( soma.x > neuronMaxX ) neuronMaxX = soma.x;
			if ( axon.x > neuronMaxX ) neuronMaxX = axon.x;
			if ( soma.y > neuronMaxY ) neuronMaxY = soma.y;
			if ( axon.y > neuronMaxY ) neuronMaxY = axon.y;
			if ( soma.z > neuronMaxZ ) neuronMaxZ = soma.z;
			if ( axon.z > neuronMaxZ ) neuronMaxZ = axon.z;
		}
	}

	double margin = 0.5;

	neuronMinX -= mNNCenter.at( 0 );
	neuronMinY -= mNNCenter.at( 1 );
	neuronMinZ -= mNNCenter.at( 2 );
	neuronMaxX -= mNNCenter.at( 0 );
	neuronMaxY -= mNNCenter.at( 1 );
	neuronMaxZ -= mNNCenter.at( 2 );

	neuronMinX -= margin;
	neuronMinY -= margin;
	neuronMinZ -= margin;
	neuronMaxX += margin;
	neuronMaxY += margin;
	neuronMaxZ += margin;

	auto corner1 = QVector3D( neuronMinX, neuronMinY, neuronMinZ );
	auto corner2 = QVector3D( neuronMinX, neuronMinY, neuronMaxZ );
	auto corner3 = QVector3D( neuronMinX, neuronMaxY, neuronMinZ );
	auto corner4 = QVector3D( neuronMinX, neuronMaxY, neuronMaxZ );
	auto corner5 = QVector3D( neuronMaxX, neuronMinY, neuronMinZ );
	auto corner6 = QVector3D( neuronMaxX, neuronMinY, neuronMaxZ );
	auto corner7 = QVector3D( neuronMaxX, neuronMaxY, neuronMinZ );
	auto corner8 = QVector3D( neuronMaxX, neuronMaxY, neuronMaxZ );

	auto lineColor = QColor( 128, 128, 128, 64 );

	drawLine( corner1, corner2, lineColor, mRootEntity, LineType::Axon );
	drawLine( corner1, corner3, lineColor, mRootEntity, LineType::Axon );
	drawLine( corner3, corner4, lineColor, mRootEntity, LineType::Axon );
	drawLine( corner2, corner4, lineColor, mRootEntity, LineType::Axon );

	drawLine( corner1, corner5, lineColor, mRootEntity, LineType::Axon );
	drawLine( corner2, corner6, lineColor, mRootEntity, LineType::Axon );
	drawLine( corner5, corner6, lineColor, mRootEntity, LineType::Axon );

	drawLine( corner5, corner7, lineColor, mRootEntity, LineType::Axon );
	drawLine( corner6, corner8, lineColor, mRootEntity, LineType::Axon );
	drawLine( corner4, corner8, lineColor, mRootEntity, LineType::Axon );
	drawLine( corner3, corner7, lineColor, mRootEntity, LineType::Axon );
	drawLine( corner7, corner8, lineColor, mRootEntity, LineType::Axon );

	instantiateLines();

	return mRootEntity;
}

//-----------------------------------------------------------------------------

void DEBINNRenderer::drawLine( const QVector3D& aStart, const QVector3D& aEnd, const QColor& aColor, Qt3DCore::QEntity* aRootEntity, LineType aLineType )
{
	Q_UNUSED( aRootEntity );
	Q_UNUSED( aLineType );

	QString colorId = QString::number( aColor.rgba(), 16 );
	Qt3DExtras::QPhongAlphaMaterial* material = mLineMaterials.value( colorId );

	if ( material == nullptr )
	{
        material = new Qt3DExtras::QPhongAlphaMaterial( &mMaterialsRootEntity );
        material->setAmbient( aColor );
        material->setDiffuse( aColor );
        material->setShininess(0);
        material->setAlpha( aColor.alpha() / 255.0f );
        material->setSourceRgbArg( Qt3DRender::QBlendEquationArguments::SourceAlpha );
        material->setDestinationRgbArg(  Qt3DRender::QBlendEquationArguments::OneMinusSourceAlpha );
        material->setSourceAlphaArg( Qt3DRender::QBlendEquationArguments::Zero );
        material->setDestinationAlphaArg( Qt3DRender::QBlendEquationArguments::One );
		material->setShareable( true );
		mLineMaterials.insert( colorId, material );
	}

	if ( !mLinesByMaterial.contains( colorId ) )
	{
		mLinesByMaterial.insert( colorId, QList< QVector3D >() << aStart << aEnd );
	}
	else
	{
		auto& pointList = mLinesByMaterial[ colorId ];
		pointList.push_back( aStart );
		pointList.push_back( aEnd );
	}
}


void DEBINNRenderer::instantiateLines()
{
	for ( auto it = mLinesByMaterial.begin(); it != mLinesByMaterial.end(); it++ )
	{
		Qt3DExtras::QPhongAlphaMaterial* material = mLineMaterials.value( it.key() );
		assert( material );
		const QList< QVector3D >& lines = it.value();

		instantiateLinesForMaterial( material, lines );
	}
}

void DEBINNRenderer::instantiateLinesForMaterial( Qt3DExtras::QPhongAlphaMaterial* aMaterial, const QList< QVector3D >& aLines )
{
	auto *geometry = new Qt3DCore::QGeometry( mRootEntity );

	// position vertices (start and end)
	QByteArray bufferBytes;
	bufferBytes.resize( 3 * aLines.size() * sizeof( float ) ); // start.x, start.y, start.end + end.x, end.y, end.z
	float *positions = reinterpret_cast< float* >( bufferBytes.data() );

	for ( const QVector3D& vec : aLines )
	{
		*positions++ = vec.x();
		*positions++ = vec.y();
		*positions++ = vec.z();
	}

	auto *buf = new Qt3DCore::QBuffer( geometry );
	buf->setData( bufferBytes );

	auto *positionAttribute = new Qt3DCore::QAttribute( geometry );
	positionAttribute->setName( Qt3DCore::QAttribute::defaultPositionAttributeName() );
	positionAttribute->setVertexBaseType( Qt3DCore::QAttribute::Float );
	positionAttribute->setVertexSize( 3 );
	positionAttribute->setAttributeType( Qt3DCore::QAttribute::VertexAttribute );
	positionAttribute->setBuffer( buf );
	positionAttribute->setByteStride( 3 * sizeof( float ) );
	positionAttribute->setCount( aLines.size() );
	geometry->addAttribute( positionAttribute ); // We add the vertices in the geometry

	// mesh
	auto *line = new Qt3DRender::QGeometryRenderer( mRootEntity );
	line->setGeometry( geometry );
	line->setPrimitiveType( Qt3DRender::QGeometryRenderer::Lines );

	// entity
	auto *lineEntity = new Qt3DCore::QEntity( mRootEntity );
	lineEntity->addComponent( line );

	lineEntity->addComponent( aMaterial );
}

//-----------------------------------------------------------------------------

void DEBINNRenderer::renderDebiNeuron( std::shared_ptr< muw::DEBINeuron > aDEBINeuron )
{
	QVector3D soma3D;
	QVector3D axon3D;

	if ( aDEBINeuron->neuronType() == muw::DEBINeuron::NeuronType::Input )
	{
		{
			auto startSoma = aDEBINeuron->soma();

			Qt3DCore::QEntity *sphereEntity = new Qt3DCore::QEntity( mRootEntity );
			
			double inputSomaRadius = mSettings.value("Renderer/Shapes/SphereEntity/Input/SomaRadius").toDouble();
			
			Qt3DCore::QTransform *sphereTransform = new Qt3DCore::QTransform;

			soma3D = QVector3D( startSoma.x - mNNCenter.at( 0 ), startSoma.y - mNNCenter.at( 1 ), startSoma.z - mNNCenter.at( 2 ) );

			QMatrix4x4 mat;
			mat.translate( soma3D );
			mat.scale( inputSomaRadius );
			sphereTransform->setMatrix( mat );

			sphereEntity->addComponent( mSphereMesh );
			sphereEntity->addComponent( sphereTransform );
			sphereEntity->addComponent( mMaterials.value( "Input/Soma" ) );
		}
		{
			auto axon = aDEBINeuron->axon();

			Qt3DCore::QEntity *sphereEntity = new Qt3DCore::QEntity( mRootEntity );
			
			double inputAxonRadius = mSettings.value("Renderer/Shapes/SphereEntity/Input/AxonRadius").toDouble();
			
			Qt3DCore::QTransform *sphereTransform = new Qt3DCore::QTransform;

			axon3D = QVector3D( axon.x - mNNCenter.at( 0 ), axon.y - mNNCenter.at( 1 ), axon.z - mNNCenter.at( 2 ) );

			QMatrix4x4 mat;
			mat.translate( axon3D );
			mat.scale( inputAxonRadius );
			sphereTransform->setMatrix( mat );

			sphereEntity->addComponent( mSphereMesh );
			sphereEntity->addComponent( sphereTransform );
			sphereEntity->addComponent( mMaterials.value( "Input/Axon" ) );
		}
		drawLine( soma3D, axon3D, mColors.value( "Input/SomaAxon" ), mRootEntity, LineType::Axon );
	}
	else if ( aDEBINeuron->neuronType() == muw::DEBINeuron::NeuronType::Hidden )
	{
		{
			auto soma = aDEBINeuron->soma();
		
			Qt3DCore::QEntity *sphereEntity = new Qt3DCore::QEntity( mRootEntity );
			
			double hiddenSomaRadius = mSettings.value("Renderer/Shapes/SphereEntity/Hidden/SomaRadius").toDouble();
			
			Qt3DCore::QTransform *sphereTransform = new Qt3DCore::QTransform;

			soma3D = QVector3D( soma.x - mNNCenter.at( 0 ), soma.y - mNNCenter.at( 1 ), soma.z - mNNCenter.at( 2 ) );

			QMatrix4x4 mat;
			mat.translate( soma3D );
			mat.scale( hiddenSomaRadius );
			sphereTransform->setMatrix( mat );

			sphereEntity->addComponent( mSphereMesh );
			sphereEntity->addComponent( sphereTransform );
			sphereEntity->addComponent( mMaterials.value( "Hidden/Soma" ) );

			auto dendrites = aDEBINeuron->dendrites();
			auto weights   = aDEBINeuron->weights();
			
			for (int idx = 0; idx < dendrites.size(); ++idx) 
			{
				auto dendrite = dendrites.at(idx);
				auto weight = weights.at(idx);
				QColor colorOfLine = mWeightPalette.paletteColor(weight);

				auto dendriteAxon = dendrite->axon();
				QVector3D dendriteAxon3D = QVector3D( dendriteAxon.x - mNNCenter.at( 0 ), dendriteAxon.y - mNNCenter.at( 1 ), dendriteAxon.z - mNNCenter.at( 2 ) );
				drawLine( dendriteAxon3D, soma3D, colorOfLine, mRootEntity, LineType::Dendrite );
			}
		}
		{
			auto axon = aDEBINeuron->axon();

			Qt3DExtras::QPhongMaterial *material = new Qt3DExtras::QPhongMaterial( mRootEntity );
			Qt3DCore::QEntity *sphereEntity = new Qt3DCore::QEntity( mRootEntity );
			
			double hiddenAxonRadius = mSettings.value("Renderer/Shapes/SphereEntity/Hidden/AxonRadius").toDouble();
			
			Qt3DCore::QTransform *sphereTransform = new Qt3DCore::QTransform;

			axon3D = QVector3D( axon.x - mNNCenter.at( 0 ), axon.y - mNNCenter.at( 1 ), axon.z - mNNCenter.at( 2 ) );

			QMatrix4x4 mat;
			mat.translate( axon3D );
			mat.scale( hiddenAxonRadius );
			sphereTransform->setMatrix( mat );

			sphereEntity->addComponent( mSphereMesh );
			sphereEntity->addComponent( sphereTransform );
			sphereEntity->addComponent( mMaterials.value( "Hidden/Axon" ) );
		}
		drawLine( soma3D, axon3D, mColors.value( "Hidden/SomaAxon" ), mRootEntity, LineType::Axon );
	}
	else if ( aDEBINeuron->neuronType() == muw::DEBINeuron::NeuronType::Output )
	{
		{
			auto soma = aDEBINeuron->soma();

			Qt3DExtras::QPhongMaterial *material = new Qt3DExtras::QPhongMaterial(mRootEntity);
			Qt3DCore::QEntity *sphereEntity = new Qt3DCore::QEntity(mRootEntity);
			
			double outputSomaRadius = mSettings.value("Renderer/Shapes/SphereEntity/Output/SomaRadius").toDouble();
			
			Qt3DCore::QTransform *sphereTransform = new Qt3DCore::QTransform;

			soma3D = QVector3D(soma.x - mNNCenter.at(0), soma.y - mNNCenter.at(1), soma.z - mNNCenter.at(2));

			QMatrix4x4 mat;
			mat.translate( soma3D );
			mat.scale( outputSomaRadius );
			sphereTransform->setMatrix( mat );

			sphereEntity->addComponent(mSphereMesh);
			sphereEntity->addComponent(sphereTransform);
			sphereEntity->addComponent(mMaterials.value("Output/Soma"));

			auto dendrites = aDEBINeuron->dendrites();
			auto weights = aDEBINeuron->weights();

			for (int idx = 0; idx < dendrites.size(); ++idx)
			{
				auto dendrite = dendrites.at(idx);
				auto weight   = weights.at(idx);
				QColor colorOfLine = mWeightPalette.paletteColor(weight);

				auto dendriteAxon = dendrite->axon();
				QVector3D dendriteAxon3D = QVector3D(dendriteAxon.x - mNNCenter.at(0), dendriteAxon.y - mNNCenter.at(1), dendriteAxon.z - mNNCenter.at(2));
				drawLine(dendriteAxon3D, soma3D, colorOfLine, mRootEntity, LineType::Dendrite);
			}
		}
		{
			auto endAxon = aDEBINeuron->axon();

			Qt3DCore::QEntity *sphereEntity = new Qt3DCore::QEntity( mRootEntity );
			
			double outputAxonRadius = mSettings.value("Renderer/Shapes/SphereEntity/Output/AxonRadius").toDouble();
			
			Qt3DCore::QTransform *sphereTransform = new Qt3DCore::QTransform;

			axon3D = QVector3D( endAxon.x - mNNCenter.at( 0 ), endAxon.y - mNNCenter.at( 1 ), endAxon.z - mNNCenter.at( 2 ) );

			QMatrix4x4 mat;
			mat.translate( axon3D );
			mat.scale( outputAxonRadius );
			sphereTransform->setMatrix( mat );

			sphereEntity->addComponent( mSphereMesh );
			sphereEntity->addComponent( sphereTransform );
			sphereEntity->addComponent( mMaterials.value( "Output/Axon" ) );
		}
		drawLine( soma3D, axon3D, mColors.value( "Output/SomaAxon" ), mRootEntity, LineType::Axon );
	}
}

//-----------------------------------------------------------------------------

}
