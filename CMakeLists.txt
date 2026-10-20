cmake_minimum_required(VERSION 3.24)

project(cdnns C CUDA)

set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

set(SRC_LIST)

set(MODULES
	elu
	# fused_linear
	leakyrelu
	linear
	relu
	sigmoid
	softmax
	tanh
)

foreach(module IN LISTS MODULES)
    	list(APPEND SRC_LIST
		src/${module}/${module}_wrapper.c
		src/${module}/${module}.c
	)
endforeach()

add_library(cdnns SHARED ${SRC_LIST})

target_include_directories(cdnns PUBLIC
	include/
)

target_sources(cdnns PRIVATE
	${SRC_LIST}
)
target_link_libraries(cdnns PRIVATE
	m
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