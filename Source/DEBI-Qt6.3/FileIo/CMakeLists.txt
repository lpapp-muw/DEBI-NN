project(FileIo)
set(sources TabularDataFileIo.cpp)

add_library(FileIo OBJECT ${sources})
target_link_libraries(FileIo Qt6::Core)
