# struct GraphvizVisitor

*Defined at src/details/ArborX_DetailsTreeVisualization.hpp#68*

 Produces node and edges statements to be listed for a graph in DOT format: ``` digraph g {   root = i0;<paste node and edges statements here> } ```



## Members

public std::ostream & _os



## Functions

### visit

*public void visit(const Tree & tree, int node)*

*Defined at src/details/ArborX_DetailsTreeVisualization.hpp#73*

### visitNode

*public void visitNode(const Tree & tree, int node)*

*Defined at src/details/ArborX_DetailsTreeVisualization.hpp#80*

### visitEdgesStartingFromNode

*public void visitEdgesStartingFromNode(const Tree & tree, int node)*

*Defined at src/details/ArborX_DetailsTreeVisualization.hpp#89*



