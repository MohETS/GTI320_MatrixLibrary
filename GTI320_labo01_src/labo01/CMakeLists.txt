cmake_minimum_required(VERSION 3.15)

project(labo01)

# Setup language requirements
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add .cpp and .h files
set(HEADERS DenseStorage.h MatrixBase.h Matrix.h Math3D.h Vector.h Operators.h)
set(SOURCE main.cpp)
add_executable(labo01 ${SOURCE} ${HEADERS})

# Add linking information for Google Test
target_link_libraries(labo01 gtest)

# Set labo01 as the startup project for Visual Studio
if( MSVC )
	set_property(TARGET labo01 PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/labo01)
endif()

