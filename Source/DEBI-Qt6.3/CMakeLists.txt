project(DEBI)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

find_package(Qt6 COMPONENTS Core Gui Widgets 3DCore 3DRender 3DExtras 3DAnimation REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(DataRepresentation)
add_subdirectory(Evaluation)
add_subdirectory(FileIo)
add_subdirectory(TestApplication)
