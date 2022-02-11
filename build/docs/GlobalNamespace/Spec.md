# struct Spec

*Defined at benchmarks/bvh_driver/benchmark_registration.hpp#26*

## Members

basic_string backends

public int n_values

public int n_queries

public int n_neighbors

public _Bool sort_predicates

public int buffer_size

public enum PointCloudType source_point_cloud_type

public enum PointCloudType target_point_cloud_type



## Functions

### Spec

*public void Spec()*

*Defined at benchmarks/bvh_driver/benchmark_registration.hpp#37*

### Spec

*public void Spec(const std::string & spec_string)*

*Defined at benchmarks/bvh_driver/benchmark_registration.hpp#38*

### create_label_construction

*public basic_string create_label_construction(const std::string & tree_name)*

*Defined at benchmarks/bvh_driver/benchmark_registration.hpp#61*

### create_label_radius_search

*public basic_string create_label_radius_search(const std::string & tree_name, const std::string & flavor)*

*Defined at benchmarks/bvh_driver/benchmark_registration.hpp#70*

### create_label_knn_search

*public basic_string create_label_knn_search(const std::string & tree_name, const std::string & flavor)*

*Defined at benchmarks/bvh_driver/benchmark_registration.hpp#84*



