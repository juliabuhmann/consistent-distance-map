all: graph_cut

graph_cut: graph.o maxflow.o graph_cut_final.o
	g++ -o graph_cut graph.o maxflow.o graph_cut_final.o

graph.o: block.o
	g++ -c graph.cpp

block.o:
	g++ -c block.h

maxflow.o: graph.o
	g++ -c maxflow.cpp

graph_cut_final.o: maxflow.o
	g++ -c -std=c++11 graph_cut_final.cpp

clean:
	rm *o graph_cut
