

find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)

add_cython_target(histogram CXX)
add_library(histogram MODULE ${histogram})
python_extension_module(histogram)
target_include_directories(histogram PRIVATE ${NUMPY_INCLUDE_DIRS})

install(TARGETS histogram LIBRARY DESTINATION skbeam/core/accumulators)
