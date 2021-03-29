add_executable(ArborX_HDBSCAN.exe
  parallel_boruvka.cpp
  main.cpp
)
set_target_properties(ArborX_HDBSCAN.exe PROPERTIES
  CXX_STANDARD 14
  CXX_EXTENSIONS OFF
)
target_link_libraries(ArborX_HDBSCAN.exe PRIVATE Boost::headers)
add_test(NAME ArborX_HDBSCAN_Example COMMAND ./ArborX_HDBSCAN.exe ${CMAKE_CURRENT_SOURCE_DIR}/notebooks/twoD-20.pts)

find_package(Boost)
add_executable(ArborX_BoostGraph.exe
  ArborX_BoostGraph.cpp
)
target_link_libraries(ArborX_BoostGraph.exe)
add_test(NAME ArborX_BoostGraph_Example COMMAND ./ArborX_BoostGraph.exe)
target_link_libraries(ArborX_BoostGraph.exe PRIVATE Boost::headers)
set_target_properties(ArborX_BoostGraph.exe PROPERTIES
  CXX_STANDARD 14
  CXX_EXTENSIONS OFF
)