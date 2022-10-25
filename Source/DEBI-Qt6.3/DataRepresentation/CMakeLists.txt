project(DataRepresentation)
set(sources
	Array2D.cpp
	TabularData.cpp
)

add_library(DataRepresentation OBJECT ${sources})
target_link_libraries(DataRepresentation Qt6::Core)
