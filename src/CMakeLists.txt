
find_package(NumPy REQUIRED)
find_package(OpenMP)

add_library(ctrans MODULE ctrans.c)
python_extension_module(ctrans)
if(OPENMP_FOUND)
  target_compile_options(ctrans PRIVATE ${OpenMP_C_FLAGS})
endif()
target_include_directories(ctrans PRIVATE ${NUMPY_INCLUDE_DIRS})

install(TARGETS ctrans LIBRARY DESTINATION skbeam/ext)
