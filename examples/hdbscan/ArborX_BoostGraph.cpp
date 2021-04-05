#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/edge_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>

#include <fstream>
#include <iostream>

template <class Graph>
void compute_minimum_spanning_tree_and_print(Graph g,
                                            std::ostream &os = std::cout)
{
  typename boost::property_map<Graph, boost::edge_weight_t>::type weight =
      get(boost::edge_weight, g);
  using Edge = typename boost::graph_traits<Graph>::edge_descriptor;
  std::vector<Edge> spanning_tree;

  boost::kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));

  std::cout << "Print the edges in the MST:" << std::endl;
  for (typename std::vector<Edge>::iterator ei = spanning_tree.begin();
       ei != spanning_tree.end(); ++ei)
  {
    std::cout << source(*ei, g) << " <--> " << target(*ei, g)
              << " with weight of " << weight[*ei] << std::endl;
  }

  os << "graph A {\n"
     << " rankdir=LR\n"
     << " size=\"3,3\"\n"
     << " ratio=\"filled\"\n"
     << " edge[style=\"bold\"]\n"
     << " node[shape=\"circle\"]\n";
  typename boost::graph_traits<Graph>::edge_iterator eiter, eiter_end;
  for (boost::tie(eiter, eiter_end) = edges(g); eiter != eiter_end; ++eiter)
  {
    os << source(*eiter, g) << " -- " << target(*eiter, g);
    if (std::find(spanning_tree.begin(), spanning_tree.end(), *eiter) !=
        spanning_tree.end())
      os << "[color=\"black\", label=\"" << get(boost::edge_weight, g, *eiter)
         << "\"];\n";
    else
      os << "[color=\"gray\", label=\"" << get(boost::edge_weight, g, *eiter)
         << "\"];\n";
  }
  os << "}\n";
}

// Convert to Boost.Graph
void convert(std::vector<std::pair<int, int>> const &edges,
             std::vector<float> const &weights, int num_nodes)
{
  using Graph = boost::adjacency_list<
      /*OutEdgeList=*/boost::vecS, /*VertexList=*/boost::vecS,
      /*Directed=*/boost::undirectedS,
      /*VertexProperties=*/boost::no_property,
      /*EdgeProperties=*/boost::property<boost::edge_weight_t, float>>;
  Graph g(edges.begin(), edges.end(), weights.begin(), num_nodes);
  boost::dynamic_properties dp;
  dp.property("weight", get(boost::edge_weight, g));
  boost::write_graphviz_dp(std::cout, g, dp, std::string("weight"));
}

int main()
{
  using Graph = boost::adjacency_list<
      /*OutEdgeList=*/boost::vecS, /*VertexList=*/boost::vecS,
      /*Directed=*/boost::undirectedS,
      /*VertexProperties=*/boost::no_property,
      /*EdgeProperties=*/boost::property<boost::edge_weight_t, int>>;
  using E = std::pair<int, int>;

  const int num_nodes = 5;
  E edge_array[] = {E(0, 2), E(1, 3), E(1, 4), E(2, 1),
                    E(2, 3), E(3, 4), E(4, 0), E(4, 1)};
  int weights[] = {1, 1, 2, 7, 3, 1, 1, 1};
  std::size_t num_edges = sizeof(edge_array) / sizeof(E);
  Graph g(edge_array, edge_array + num_edges, weights, num_nodes);

  compute_minimum_spanning_tree_and_print(g);

  return EXIT_SUCCESS;
}