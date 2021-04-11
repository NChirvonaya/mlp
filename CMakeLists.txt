cmake_minimum_required(VERSION 3.0)
project(mlp)

set(SOURCES 
src/main.cpp
src/dataitem.cpp
src/layer.cpp
src/neuron.cpp
src/weights.cpp)

set(HEADERS
src/dataitem.h
src/layer.h
src/neuron.h
src/weights.h)

add_executable(mlp ${SOURCES} ${HEADERS})