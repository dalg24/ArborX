set(ArborX_TARGET ArborX)

# We require version 1.4.0 or higher but the format used by Google benchmark is
# wrong and thus, we cannot check the version during the configuration step.
find_package(benchmark REQUIRED)

find_package(Threads REQUIRED)

add_executable(ArborX_OccupancyCalculator.exe main.cpp)
target_link_libraries(ArborX_OccupancyCalculator.exe ${ArborX_TARGET} benchmark::benchmark Threads::Threads)
add_test(NAME ArborX_OccupancyCalculator_Benchmark COMMAND ./ArborX_OccupancyCalculator.exe --benchmark_color=true)
