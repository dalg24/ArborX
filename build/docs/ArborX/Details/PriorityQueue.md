# class PriorityQueue

*Defined at src/details/ArborX_DetailsPriorityQueue.hpp#31*

## Members

private Container _c

private Compare _compare



## Functions

### PriorityQueue<T, Compare, Container>

*public void PriorityQueue<T, Compare, Container>()*

*Defined at src/details/ArborX_DetailsPriorityQueue.hpp#45*

### PriorityQueue<T, Compare, Container>

*public void PriorityQueue<T, Compare, Container>(const Container & c)*

*Defined at src/details/ArborX_DetailsPriorityQueue.hpp#47*

### empty

*public _Bool empty()*

*Defined at src/details/ArborX_DetailsPriorityQueue.hpp#54*

 Capacity

### size

*public ArborX::Details::PriorityQueue::size_type size()*

*Defined at src/details/ArborX_DetailsPriorityQueue.hpp#55*

### top

*public ArborX::Details::PriorityQueue::reference top()*

*Defined at src/details/ArborX_DetailsPriorityQueue.hpp#58*

 Element access

### top

*public ArborX::Details::PriorityQueue::const_reference top()*

*Defined at src/details/ArborX_DetailsPriorityQueue.hpp#59*

### push

*public void push(const ArborX::Details::PriorityQueue::value_type & value)*

*Defined at src/details/ArborX_DetailsPriorityQueue.hpp#62*

 Modifiers

### push

*public void push(ArborX::Details::PriorityQueue::value_type && value)*

*Defined at src/details/ArborX_DetailsPriorityQueue.hpp#67*

### emplace

*public void emplace(Args &&... args)*

*Defined at src/details/ArborX_DetailsPriorityQueue.hpp#73*

### pop

*public void pop()*

*Defined at src/details/ArborX_DetailsPriorityQueue.hpp#78*

### popPush

*public void popPush(Args &&... args)*

*Defined at src/details/ArborX_DetailsPriorityQueue.hpp#87*

 in TreeTraversal::nearestQuery, pop() is often followed by push which is an opportunity for doing a single bubble-down operation instead of paying for both one bubble-down and one bubble-up

### data

*public typename Container::pointer data()*

*Defined at src/details/ArborX_DetailsPriorityQueue.hpp#96*

 Accessors that shouldn't be there but that are convenient in TreeTraversal::nearestQuery()

### valueComp

*public const ArborX::Details::PriorityQueue::value_compare & valueComp()*

*Defined at src/details/ArborX_DetailsPriorityQueue.hpp#100*



