# class CmdLineArgs

*Defined at benchmarks/bvh_driver/bvh_driver.cpp#156*

 NOTE Motivation for this class that stores the argument count and values is I could not figure out how to make the parser consume arguments with Boost.Program_options Benchmark removes its own arguments from the command line arguments. This means, that by virtue of returning references to internal data members in argc() and argv() function, it will necessarily modify the members. It will decrease _argc, and "reduce" _argv data. Hence, we must keep a copy of _argv that is not modified from the outside to release memory in the destructor correctly.



## Members

private int _argc

vector _argv

vector _owner_ptrs



## Functions

### CmdLineArgs

*public void CmdLineArgs(const std::vector<std::string> & args, const char * exe)*

*Defined at benchmarks/bvh_driver/bvh_driver.cpp#164*

### ~CmdLineArgs

*public void ~CmdLineArgs()*

*Defined at benchmarks/bvh_driver/bvh_driver.cpp#178*

### argc

*public int & argc()*

*Defined at benchmarks/bvh_driver/bvh_driver.cpp#186*

### argv

*public char ** argv()*

*Defined at benchmarks/bvh_driver/bvh_driver.cpp#188*



