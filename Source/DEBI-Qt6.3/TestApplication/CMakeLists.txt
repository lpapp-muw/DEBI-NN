project(TestApplication)

set(sources
	DEBINNCrossValidator.cpp
	DEBINNRenderController.cpp
	main.cpp
	# TestApplication.cpp
	TestApplication.qrc
)

add_executable(TestApplication ${sources})
target_link_libraries(TestApplication Qt6::Core Qt6::Gui Qt6::Widgets Qt6::3DCore Qt6::3DExtras Qt6::3DAnimation DataRepresentation Evaluation FileIo)
