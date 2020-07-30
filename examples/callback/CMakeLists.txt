add_executable(ArborX_Callback.exe example_callback.cpp)
target_link_libraries(ArborX_Callback.exe ArborX::ArborX)
add_test(NAME ArborX_Callback_Example COMMAND ./ArborX_Callback.exe)

if(ARBORX_ENABLE_MPI)
  add_executable(ArborX_DistributedTree_Callback.exe example_mpi_callback.cpp)
  target_link_libraries(ArborX_DistributedTree_Callback.exe ArborX::ArborX)
  add_test(NAME ArborX_DistributedTree_Callback_Example COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./ArborX_DistributedTree_Callback.exe ${MPIEXEC_POSTFLAGS})
endif()
