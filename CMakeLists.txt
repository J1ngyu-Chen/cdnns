cmake_minimum_required(VERSION 3.24)

project(cdnns C CUDA)

set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# CUDA API
include_directories(/usr/local/cuda-13.0/include)
link_libraries(/usr/local/cuda-13.0/lib64/libcudart.so)

foreach(sub_lib
	elu
	leakyrelu
	linear
	relu
	sigmoid
	softmax
	tanh
)
	file(GLOB_RECURSE SUB_LIB_SRCS CONFIGURE_DEPENDS
		libs/${sub_lib}/*.c
		libs/${sub_lib}/*.cu
	)
	add_library(${sub_lib} OBJECT
		${SUB_LIB_SRCS}
	)
	target_include_directories(${sub_lib} PRIVATE
		include
		libs/${sub_lib}
	)
	target_compile_definitions(${sub_lib} PRIVATE
		#-D IMPL_BASIC
		-D IMPL_CUDA
	)
    	list(APPEND OBJECT_LIST $<TARGET_OBJECTS:${sub_lib}>)
endforeach()

add_library(cdnns SHARED ${OBJECT_LIST})
set_target_properties(cdnns PROPERTIES 
	PUBLIC_HEADER include/cdnns.h
)

install(TARGETS cdnns
	LIBRARY DESTINATION /usr/local/lib/
	PUBLIC_HEADER DESTINATION /usr/loacl/include/
)

#[[
add_executable(test
    	test_bench.c
)

target_link_libraries(test PRIVATE
	cdnns
)
]]