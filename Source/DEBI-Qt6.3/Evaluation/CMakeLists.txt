project(Evaluation)
set(sources
	AbstractAnalytics.cpp
	AbstractModel.cpp
	ConfusionMatrixAnalytics.cpp
	DataPackage.cpp
	DEBINeuron.cpp
	DEBINN.cpp
	DEBINNFactory.cpp
	DEBINNRenderer.cpp
	GeneticAlgorithmOptimizer.cpp
	LossAnalytics.cpp
	MultiSubsetFoldGenerator.cpp
	Palette.cpp
	TabularDataFilter.cpp
)

add_library(Evaluation OBJECT ${sources})
target_link_libraries(Evaluation Qt6::Core Qt6::Gui Qt6::3DRender)
