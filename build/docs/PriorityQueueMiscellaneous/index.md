# namespace PriorityQueueMiscellaneous



## Records

* [pop_push](pop_push.md)
* [pop_push_id](pop_push_id.md)
* [maintain_heap_properties](maintain_heap_properties.md)
* [maintain_heap_properties_id](maintain_heap_properties_id.md)


## Functions

### check_heap

*void check_heap(const PriorityQueue & queue, const std::vector<typename PriorityQueue::value_type> & heap_ref)*

*Defined at test/tstPriorityQueueMiscellaneous.cpp#28*

 NOTE The tests below check that the priority queue invariant is maintained while inserting and removing elements into the queue.  They rely on a hack (reinterpret_cast) to access the underlying container.

### pop_push_invoker

*void pop_push_invoker()*

*Defined at test/tstPriorityQueueMiscellaneous.cpp#42*

### check_heap

*void check_heap(const PriorityQueue & queue)*

*Defined at test/tstPriorityQueueMiscellaneous.cpp#73*

### maintain_heap_properties_invoker

*void maintain_heap_properties_invoker()*

*Defined at test/tstPriorityQueueMiscellaneous.cpp#90*



