#include <stdio.h>
#include "graph.h"
#include <fstream>
#include <sstream>
#include <string.h>
#include <iostream>
#include <ctime>
#include <limits>


int graph_cut3D(std::string input, std::string output)
{
        
    
    printf("#####################################\n      Solving Graph-Cut Problem\n#####################################\n\n");
    std::clock_t    start, init, inter1, inter2, graph, optimiz;
    start = std::clock();
    int imax = std::numeric_limits<int>::max();
    float fmax = std::numeric_limits<float>::max();
    
    float INF = fmax/1024;



    // ------------------------------------------------------------------------
    // 1. Read graph from file
    // ------------------------------------------------------------------------

    // File has the following structure:
    //
    //  0. Any line starting with '#' is ignored.
    //
    //  1. First line contains space-separated integers, in order:
    //      a. the number of nodes (1 int)
    //      b. the maximal distance (1 int)
    //      c. the dimensions of the graph (call it d) (1 int)
    //      d. the shape of the graph (d ints)
    //      e. the shape of the neighborhood structure (d ints)
    //
    //  2. Then a line of space-separated integers that represents the
    //     neighborhood structure. This structure is a d-D array with odd
    //     dimensions where non-zero entries represent existing edges.
    //
    //  3. Lastly a line of space-separated floats that represents the unaries
    //     of the graph.
    //
    
    // Open file as stream.
    std::ifstream infile(input);
    std::string line;
    
    
    
    int n_nodes, max_dist, dims;
    
    // Read lines until the first not commented line.
    while(std::getline(infile, line)){
        if (line[0] != '#')
            break;
    }
    
    // Assign first 3 integers in the corresponding variables.
    std::istringstream iss(line);
    iss >> n_nodes >> max_dist >> dims;
    
    // Declare and assign array of corresponding dimension.
    int* shape_weights = new int[dims];
    for (int i=0; i<dims; i++)
        iss >> shape_weights[i];
        
    printf("Building graph:\n    Number of nodes: %d\n    Maximal distance: %d\n    Graph dimensions: %dx%dx%dx%d\n",n_nodes, max_dist, shape_weights[0], shape_weights[1], shape_weights[2], max_dist);
    
    
    
    int* shape_neigh = new int[dims];
    for (int i=0; i<dims; i++)
        iss >> shape_neigh[i];
    
    iss.clear();//clear any bits set
    iss.str(std::string());

    
    
    
    
    
    
    
    // 2nd non-commented line is the neighborhood-structure data.
    int**** neighborhood = new int***[shape_neigh[0]];
    for(int i = 0; i < shape_neigh[0]; ++i){
        neighborhood[i] = new int**[shape_neigh[1]];
        for(int j = 0; j < shape_neigh[1]; ++j){
            neighborhood[i][j] = new int*[shape_neigh[2]];
            for(int k = 0; k < shape_neigh[2]; ++k)
                neighborhood[i][j][k] = new int[shape_neigh[3]];
        }
    }
    while(std::getline(infile, line)){
        if (line[0] != '#')
            break;
    }
    iss.clear();//clear any bits set
    iss.str(std::string());
    iss.str(line);
    int n_neighbors = 0;
    for(int i = 0; i < shape_neigh[0]; ++i)
        for(int j = 0; j < shape_neigh[1]; ++j)
            for(int k = 0; k < shape_neigh[2]; ++k)
                for(int l = 0; l < shape_neigh[3]; ++l){
                    iss >> neighborhood[i][j][k][l];
                    n_neighbors += neighborhood[i][j][k][l];}  
    
    
    
    
    // 3rd non-commented line is the weights data.
    float**** weights = new float***[shape_weights[0]];
    for(int i = 0; i < shape_weights[0]; ++i){
        weights[i] = new float**[shape_weights[1]];
        for(int j = 0; j < shape_weights[1]; ++j){
                weights[i][j] = new float*[shape_weights[2]];
                for(int k = 0; k < shape_weights[2]; ++k)
                    weights[i][j][k] = new float[shape_weights[3]];
        }
    }
    
    while(std::getline(infile, line)){
        if (line[0] != '#')
            break;
    }
    

    iss.clear();//clear any bits set
    iss.str(std::string());
    iss.str(line);
    for(int i = 0; i < shape_weights[0]; ++i)
        for(int j = 0; j < shape_weights[1]; ++j)
            for(int k = 0; k < shape_weights[2]; ++k)
                for(int l = 0; l < shape_weights[3]; ++l){
                    float my_number;
                    iss >> my_number;//weights[i][j][k];
                    weights[i][j][k][l] = my_number;
                
            }
                    
    
    
    
    // BUILD GRAPH WITH NODES
    bool repeat = true;
    int count = 0;
    bool second_iteration = false;
    
    init = std::clock();
    while (repeat){
        inter1 = std::clock();
        
        typedef Graph<float,float,float> GraphType;

        GraphType *g = new GraphType(/*estimated # of nodes*/ n_nodes, /*estimated # of edges*/ n_neighbors*n_nodes); 
        
        for (int i=0; i<n_nodes; ++i)
            g -> add_node();
            
        
        
        
        
        
        // ADD INTERNAL EDGES
        
        int yzw = shape_weights[1]*shape_weights[2]*shape_weights[3];
        int zw = shape_weights[2]*shape_weights[3];
        int w_row = shape_weights[3];
        int ind;
        int* half_shape = new int[dims];
        int edge_count = 0;
        int tedge_count = 0;
        half_shape[0] = (shape_neigh[0]-1)/2;
        half_shape[1] = (shape_neigh[1]-1)/2;
        half_shape[2] = (shape_neigh[2]-1)/2;
        half_shape[3] = (shape_neigh[3]-1)/2;
        for (int x=0; x < shape_weights[0]; ++x){
            for (int y=0; y < shape_weights[1]; ++y){
                for (int z=0; z < shape_weights[2]; ++z){
                    for (int w=0; w < shape_weights[3]; ++w){
                    
                        int ind_src = yzw*x + zw*y + z*w_row + w;
                        
                        for (int sx=0; sx < shape_neigh[0]; ++sx){
                            for (int sy=0; sy < shape_neigh[1]; ++sy){
                                for (int sz=0; sz < shape_neigh[2]; ++sz){
                                    for (int sw=0; sw < shape_neigh[3]; ++sw){
                                    
                                        int px,py,pz,pw;
                                        px = sx-half_shape[0];
                                        py = sy-half_shape[1];
                                        pz = sz-half_shape[2];
                                        pw = sw-half_shape[3];
                                        
                                        
                                        //printf ("%d, %d, %d\n", sx, sy, sz);;
                                        if (neighborhood[sx][sy][sz][sw] != 0){
                                        
                                            if (x+px >= 0 && x+px < shape_weights[0] && y+py >= 0 && y+py < shape_weights[1] && z+pz >= 0 && z+pz < shape_weights[2] && w+pw >= 0 && w+pw < shape_weights[3]){
                                                
                                                int ind_dest = (x+px)*yzw + (y+py)*zw + (z+pz)*w_row + w+pw;
                                                
                                                //printf ("INF: %d, %d, %d, %d  ->  ", x, y, z, w);
                                                //printf ("%d, %d, %d, %d\n", x+px, y+py, z+pz, w+pw);
                                                //~ printf ("INF: %d, %d, %d  ->  ", x, y, w);
                                                //~ printf ("%d, %d, %d\n", x+px, y+py, w+pw);
                                                //printf ("%d -> %d\n", ind_src, ind_dest);
                                                
                                                g -> add_edge( ind_src, ind_dest,    /* capacities */  INF, 0 );
                                                edge_count++;
                                            }
                                            else
                                                if (x+px >= 0 && x+px < shape_weights[0] && y+py >= 0 && y+py < shape_weights[1] && z+pz >= 0 && z+pz < shape_weights[2] && w+pw < 0){
                                                    
                                                    int ind_dest = (x+px)*yzw + (y+py)*zw + (z+pz)*w_row + 0;
                                                    
                                                    //printf ("%d -> %d\n", ind_src, ind_dest);
                                                    if (ind_src != ind_dest){
                                                        //~ printf ("INF: %d, %d, %d  ->  ", x, y, w);
                                                        //~ //printf ("INF: %d, %d, %d, %d  ->  ", x, y, z, w);
                                                        //~ //printf ("%d, %d  ->  %d, %d\n", x, y, (x+px), (y+py));
                                                        //~ //printf ("%d, %d, %d, %d\n", x+px, y+py, z+pz, 0);
                                                        //~ printf ("%d, %d, %d\n", x+px, y+py, 0);
                                                        g -> add_edge( ind_src, ind_dest,    /* capacities */  INF, 0 );
                                                        edge_count++;
                                                    }
                                                }
                                            }
                                    }
                                }
                            }
                        }
                                        
                        
                        if (weights[x][y][z][w] <= 0){
                            
                            //printf ("%f, 0: %d, %d, %d, %d  ->  source\n", -weights[x][y][z][w], x, y, z, w);
                            //~ printf ("%f, 0: %d, %d, %d  ->  source\n", -weights[x][y][z][w], x, y, w);
                            g -> add_tweights( ind_src,    /* capacities */   float(-weights[x][y][z][w]), float(0.));
                            tedge_count++;}
                        else{
                            //printf ("0, %f: %d, %d, %d, %d  ->  sink\n", weights[x][y][z][w], x, y, z, w);
                            //~ printf ("0, %f: %d, %d, %d  ->  sink\n", weights[x][y][z][w], x, y, w);
                            g -> add_tweights( ind_src,    /* capacities */   float(0.), float(weights[x][y][z][w]));
                            tedge_count++;}
                    }
                }
            }
        }
        

        
        
        
        
        if (!second_iteration)
            printf("    Number of internal edges: %d\n    Number of terminal edges: %d\n\n", edge_count, tedge_count);
        
        printf("Starting optimization:\n");
        
        graph = std::clock();
        inter2 = std::clock();
        int flow;
        flow = g -> maxflow();
        //std::cout << "\nFlow:" << std::to_string(flow) << "\n";
        optimiz = std::clock();
        
//        std::cout << "Optimization done in " << (optimiz - inter2) / (double)(CLOCKS_PER_SEC / 1000000) << " s. Flow: " << std::to_string(flow) << std::endl;
        std::cout << "    Optimization done in " << (optimiz - inter2) / (double)(CLOCKS_PER_SEC) << " s.\n    Flow: " << std::to_string(flow) << std::endl << std::endl;
        
        bool flag = true;
        
        for (int i = 0; i < n_nodes; i++){
            if (g->what_segment(i) == GraphType::SOURCE){
                flag = false;
                //std::cout << std::to_string(i) << "Quitting\n";
                }
            }
        
        
        
        
        
        if (flag){
            std::cout << "Empty solution found. Restarting optimization with translated graph.\n\n";
            repeat = true;
            if (count > 0)
                break;
            count++;
            float sum = 1.;
            for (int x=0; x < shape_weights[0]; ++x){
                for (int y=0; y < shape_weights[1]; ++y){
                    for (int z=0; z < shape_weights[2]; ++z){
                        sum += weights[x][y][z][0];
                    }
                }
            }
            
            weights[0][0][0][0] -= sum;
            
            second_iteration = true;
            
    
        }
        else{
            repeat = false;

                    

            std::cout << "Writing results to file." << std::endl;
            std::ofstream myfile (output);
            if (myfile.is_open())
              {
                myfile << std::to_string(g->what_segment(0) == GraphType::SOURCE) ;
                for(int i = 1; i < n_nodes; i++){
                    myfile << " " << std::to_string(g->what_segment(i) == GraphType::SOURCE) ;
                }
                myfile.close();
              }

            }
            
        if (second_iteration)
            delete g;
    }
    
    infile.close();
	return 0;
}







int graph_cut3D_VCE(std::string input, std::string output)
{
        
    printf("#####################################\n      Solving Graph-Cut Problem\n#####################################\n\n");
    std::clock_t    start, init, inter1, inter2, graph, optimiz;
    start = std::clock();
    int imax = std::numeric_limits<int>::max();
    float fmax = std::numeric_limits<float>::max();
    
    float INF = fmax/1024;



    // ------------------------------------------------------------------------
    // 1. Read graph from file
    // ------------------------------------------------------------------------

    // File has the following structure:
    //
    //  0. Any line starting with '#' is ignored.
    //
    //  1. First line contains space-separated integers, in order:
    //      a. the number of nodes (1 int)
    //      b. the maximal distance (1 int)
    //      c. the dimensions of the graph (call it d) (1 int)
    //      d. the shape of the graph (d ints)
    //      e. the shape of the neighborhood structure (d ints)
    //
    //  2. Then a line of space-separated integers that represents the
    //     neighborhood structure. This structure is a d-D array with odd
    //     dimensions where non-zero entries represent existing edges.
    //
    //  3. Lastly a line of space-separated floats that represents the unaries
    //     of the graph.
    //
    
    // Open file as stream.
    std::ifstream infile(input);
    std::string line;
    
    
    
    int n_nodes, max_dist, dims;
    
    // Read lines until the first not commented line.
    while(std::getline(infile, line)){
        if (line[0] != '#')
            break;
    }
    
    // Assign first 3 integers in the corresponding variables.
    std::istringstream iss(line);
    iss >> n_nodes >> max_dist >> dims;
    
    // Declare and assign array of corresponding dimension.
    int* shape_weights = new int[dims];
    for (int i=0; i<dims; i++)
        iss >> shape_weights[i];
        
    printf("Building graph:\n    Number of nodes: %d\n    Maximal distance: %d\n    Graph dimensions: %dx%dx%dx%d\n",n_nodes, max_dist, shape_weights[0], shape_weights[1], shape_weights[2], max_dist);
    
    
    
    int* shape_neigh = new int[dims];
    for (int i=0; i<dims; i++)
        iss >> shape_neigh[i];
    
    iss.clear();//clear any bits set
    iss.str(std::string());

    
    
    
    
    
    // ----------------------------------------------------------
    // 2nd non-commented line is the neighborhood-structure data.
    // ----------------------------------------------------------
    
    int**** neighborhood = new int***[shape_neigh[0]];
    for(int i = 0; i < shape_neigh[0]; ++i){
        neighborhood[i] = new int**[shape_neigh[1]];
        for(int j = 0; j < shape_neigh[1]; ++j){
            neighborhood[i][j] = new int*[shape_neigh[2]];
            for(int k = 0; k < shape_neigh[2]; ++k)
                neighborhood[i][j][k] = new int[shape_neigh[3]];
        }
    }
    while(std::getline(infile, line)){
        if (line[0] != '#')
            break;
    }
    iss.clear();//clear any bits set
    iss.str(std::string());
    iss.str(line);
    int n_neighbors = 0;
    for(int i = 0; i < shape_neigh[0]; ++i)
        for(int j = 0; j < shape_neigh[1]; ++j)
            for(int k = 0; k < shape_neigh[2]; ++k)
                for(int l = 0; l < shape_neigh[3]; ++l){
                    iss >> neighborhood[i][j][k][l];
                    n_neighbors += neighborhood[i][j][k][l];}  
    
    
    //~ printf("Neighborhood: ");
    //~ printf(line.c_str());
    //~ printf("\n");
    
    
    // ----------------------------------------------------------
    // 3rd non-commented line is the unaries cost matrix.
    // ----------------------------------------------------------
    
    int col_height = max_dist+1;
    float** unaries_cost_mat = new float*[col_height];
    for(int i = 0; i < col_height; ++i)
        unaries_cost_mat[i] = new float[col_height];
        
    while(std::getline(infile, line)){
        if (line[0] != '#')
            break;
    }
    iss.clear();//clear any bits set
    iss.str(std::string());
    iss.str(line);
    for(int i = 0; i < col_height; ++i)
        for(int j = 0; j < col_height; ++j){
            iss >> unaries_cost_mat[i][j];
        }
        
    //~ printf("Unaries: ");
    //~ printf(line.c_str());
    //~ printf("\n");
    
    // ----------------------------------------------------------
    // On the next non-commented line are the binaries.
    // ----------------------------------------------------------
    while(std::getline(infile, line)){
        if (line[0] != '#'){
            break;
        }
    }
    //~ printf("Prediction: ");
    //~ printf(line.c_str());
    //~ printf("\n");    
    bool add_binaries = false;


    float**** binaries_cost_mat = new float***[col_height];
    for(int i = 0; i < col_height; ++i){
        binaries_cost_mat[i] = new float**[col_height];
        for(int j = 0; j < col_height; ++j){
            binaries_cost_mat[i][j] = new float*[col_height];
            for(int k = 0; k < col_height; ++k){
                binaries_cost_mat[i][j][k] = new float[col_height];
                for(int l = 0; l < col_height; ++l){
                    binaries_cost_mat[i][j][k][l] = 0;
                }
            }
        }
    }


    printf(line.c_str());
    if (strncmp(line.c_str(), "binaries", strlen("binaries")) == 0){
        std::getline(infile, line);
        iss.clear();//clear any bits set
        iss.str(std::string());
        iss.str(line);
    
    

            
        
        iss.clear();//clear any bits set
        iss.str(std::string());
        iss.str(line);
        for(int i = 0; i < col_height; ++i)
            for(int j = 0; j < col_height; ++j)
                for(int k = 0; k < col_height; ++k)   
                    for(int l = 0; l < col_height; ++l)
                        iss >> binaries_cost_mat[i][j][k][l];    
            
        add_binaries = true;
        //~ printf("Binaries: ");
        //~ printf(line.c_str());
        //~ printf("\n");
            
        while(std::getline(infile, line)){
            if (line[0] != '#')
                break;
        }
        
    
    }
    
    // ----------------------------------------------------------
    // Next non-commented line is the prediction data.
    // ----------------------------------------------------------
    
    //~ float**** weights = new float***[shape_weights[0]];
    //~ for(int i = 0; i < shape_weights[0]; ++i){
        //~ weights[i] = new float**[shape_weights[1]];
        //~ for(int j = 0; j < shape_weights[1]; ++j){
                //~ weights[i][j] = new float*[shape_weights[2]];
                //~ for(int k = 0; k < shape_weights[2]; ++k)
                    //~ weights[i][j][k] = new float[shape_weights[3]];
        //~ }
    //~ }
    int*** weights = new int**[shape_weights[0]];
    for(int i = 0; i < shape_weights[0]; ++i){
        weights[i] = new int*[shape_weights[1]];
        for(int j = 0; j < shape_weights[1]; ++j)
            weights[i][j] = new int[shape_weights[2]];
    }
    
    

    iss.clear();//clear any bits set
    iss.str(std::string());
    iss.str(line);
    printf("%d, %d, %d\n", shape_weights[0], shape_weights[1], shape_weights[2]);
    for(int i = 0; i < shape_weights[0]; ++i)
        for(int j = 0; j < shape_weights[1]; ++j)
            for(int k = 0; k < shape_weights[2]; ++k){
                iss >> weights[i][j][k];
                printf("%d",weights[i][j][k]);
            }
                
    printf("Prediction: ");
    printf(line.c_str());
    printf("\n");
                    
    
    
    
    // BUILD GRAPH WITH NODES
    bool repeat = true;
    int count = 0;
    bool second_iteration = false;
    
    float offset = 0;
    
    init = std::clock();
    while (repeat){
        inter1 = std::clock();
        
        typedef Graph<float,float,float> GraphType;

        GraphType *g = new GraphType(/*estimated # of nodes*/ n_nodes, /*estimated # of edges*/ n_neighbors*n_nodes); 
        
        for (int i=0; i<n_nodes; ++i)
            g -> add_node();
            
        
        
        
        
        
        // ADD INTERNAL EDGES
        
        int yzw = shape_weights[1]*shape_weights[2]*shape_weights[3];
        int zw = shape_weights[2]*shape_weights[3];
        int w_row = shape_weights[3];
        int ind;
        int* half_shape = new int[dims];
        int edge_count = 0;
        int tedge_count = 0;
        int predicted_height;
        float unary_cost;
        half_shape[0] = (shape_neigh[0]-1)/2;
        half_shape[1] = (shape_neigh[1]-1)/2;
        half_shape[2] = (shape_neigh[2]-1)/2;
        half_shape[3] = (shape_neigh[3]-1)/2;
        for (int x=0; x < shape_weights[0]; ++x){
            for (int y=0; y < shape_weights[1]; ++y){
                for (int z=0; z < shape_weights[2]; ++z){
                    
                    predicted_height = weights[x][y][z];
                    //~ iss >> weight;
                    
                    for (int w=0; w < shape_weights[3]; ++w){
                    
                        int ind_src = yzw*x + zw*y + z*w_row + w;
                        
                        
                        for (int sx=0; sx < shape_neigh[0]; ++sx){
                            for (int sy=0; sy < shape_neigh[1]; ++sy){
                                for (int sz=0; sz < shape_neigh[2]; ++sz){
                                    for (int sw=0; sw < shape_neigh[3]; ++sw){
                                        int px,py,pz,pw;
                                        px = sx-half_shape[0];
                                        py = sy-half_shape[1];
                                        pz = sz-half_shape[2];
                                        pw = sw-half_shape[3];
                                        
                                        
                                        //printf ("%d, %d, %d\n", sx, sy, sz);;
                                        if (neighborhood[sx][sy][sz][sw] != 0){
                                            if (x+px >= 0 && x+px < shape_weights[0] && y+py >= 0 && y+py < shape_weights[1] && z+pz >= 0 && z+pz < shape_weights[2] && w+pw >= 0 && w+pw < shape_weights[3]){
                                                
                                                int ind_dest = (x+px)*yzw + (y+py)*zw + (z+pz)*w_row + w+pw;
                                                
                                                //~ //printf ("INF: %d, %d, %d, %d  ->  ", x, y, z, w);
                                                //~ //printf ("%d, %d, %d, %d\n", x+px, y+py, z+pz, w+pw);
                                                //~ printf ("INF: %d, %d, %d  ->  ", x, y, w);
                                                //~ printf ("%d, %d, %d\n", x+px, y+py, w+pw);
                                                //~ //printf ("%d -> %d\n", ind_src, ind_dest);
                                                
                                                g -> add_edge( ind_src, ind_dest,    /* capacities */  INF, 0 );
                                                edge_count++;
                                                
                                                if (add_binaries && pw < 0 && (px || py || pz)){
                                                    
                                                    float binary;
                                                    int predicted_height2;
                                                    for (int h = 0; h > pw; --h){
                                                    
                                        
                                                        ind_dest = (x+px)*yzw + (y+py)*zw + (z+pz)*w_row + w+h;
                                                        predicted_height2 = weights[x+px][y+py][z+pz];
                                                        if (h == 0){
                                                            binary = binaries_cost_mat[predicted_height][predicted_height2][w][w+h-1]-binaries_cost_mat[predicted_height][predicted_height2][w][w+h];
                                                        }
                                                        else{
                                                            binary = binaries_cost_mat[predicted_height][predicted_height2][w][w+h-2] - binaries_cost_mat[predicted_height][predicted_height2][w][w+h];
                                                        
                                                        }
                                                        //~ printf ("bin:%f: %d, %d, %d  ->  %d, %d, %d\n\n", binary, x, y, w, (x+px), (y+py), (w+h));
                                                        
                                        
                                                        g -> add_edge( ind_src, ind_dest,    /* capacities */  binary, 0 );
                                                       
                                                    
                                                    }
                                                    
                                                }
                                                
                                            }
                                            else
                                                if (x+px >= 0 && x+px < shape_weights[0] && y+py >= 0 && y+py < shape_weights[1] && z+pz >= 0 && z+pz < shape_weights[2] && w+pw < 0){
                                                    
                                                    int ind_dest = (x+px)*yzw + (y+py)*zw + (z+pz)*w_row + 0;
                                                    
                                                    //printf ("%d -> %d\n", ind_src, ind_dest);
                                                    if (ind_src != ind_dest){
                                                      
                                                        //~ printf ("INF: %d, %d, %d  ->  ", x, y, w);
                                                        //printf ("INF: %d, %d, %d, %d  ->  ", x, y, z, w);
                                                        //printf ("%d, %d  ->  %d, %d\n", x, y, (x+px), (y+py));
                                                        //printf ("%d, %d, %d, %d\n", x+px, y+py, z+pz, 0);
                                                        //~ printf ("%d, %d, %d\n", x+px, y+py, 0);
                                                        g -> add_edge( ind_src, ind_dest,    /* capacities */  INF, 0 );
                                                        edge_count++;
                                                    }
                                                }
                                            }
                                            
                                    }
                                }
                            }
                        }
                                        
                        
                        //~ if (w == 0)
                            //~ unary_cost = unaries_cost_mat[predicted_height][w];
                        //~ else
                            //~ unary_cost = unaries_cost_mat[predicted_height][w] - unaries_cost_mat[predicted_height][w-1] ;
                        unary_cost = unaries_cost_mat[predicted_height][w];    
                        if (x == 0 && y == 0 && z == 0 && w == 0 && second_iteration)
                            unary_cost -= offset;
                        if (unary_cost <= 0){
                            
                            //printf ("%f, 0: %d, %d, %d, %d  ->  source\n", -weights[x][y][z][w], x, y, z, w);
                            //~ printf ("Costs for pred = %d: ", predicted_height);
                            for (int q=0;q<shape_weights[3];++q)
                                //~ printf("%d: %f, ", q, unaries_cost_mat[predicted_height][q]);
                            //~ printf("\n");
                            //~ printf ("Pred: %d; %f, 0: %d, %d, %d  ->  source\n", predicted_height, -unary_cost, x, y, w);
                            g -> add_tweights( ind_src,    /* capacities */   float(-unary_cost), float(0.));
                            tedge_count++;}
                        else{
                            //printf ("0, %f: %d, %d, %d, %d  ->  sink\n", weights[x][y][z][w], x, y, z, w);
                            //~ printf ("Pred: %d; 0, %f: %d, %d, %d  ->  sink\n", predicted_height, unary_cost, x, y, w);
                            g -> add_tweights( ind_src,    /* capacities */   float(0.), float(unary_cost));
                            tedge_count++;}
                    }
                }
            }
        }
        
        
        
        // --------------------------------------------------------------------
        // Binary costs
        // --------------------------------------------------------------------
        
        //~ while(std::getline(infile, line)){
            //~ if (strcmp(line, "# binaries list\n")==0){
                
                //~ std::getline(infile, line) // Get next line
                //~ iss.clear();//clear any bits set
                //~ iss.str(std::string());
                //~ iss.str(line);
                //~ for(int i = 0; i < shape_weights[0]; ++i)
                    //~ for(int j = 0; j < shape_weights[1]; ++j)
                        //~ for(int k = 0; k < shape_weights[2]; ++k)
                            //~ for(int l = 0; l < shape_weights[3]; ++l)
                                //~ for(int l = 0; l < shape_weights[4]; ++l){
                                    //~ float my_number;
                                    //~ iss >> my_number;//weights[i][j][k];
                                    //~ weights[i][j][k][l] = my_number;
                
                    //~ }
        //~ }
        
        
        
        
        if (!second_iteration)
            printf("    Number of internal edges: %d\n    Number of terminal edges: %d\n\n", edge_count, tedge_count);
        
        printf("Starting optimization:\n");
        
        graph = std::clock();
        inter2 = std::clock();
        int flow;
        flow = g -> maxflow();
        //std::cout << "\nFlow:" << std::to_string(flow) << "\n";
        optimiz = std::clock();
        
//        std::cout << "Optimization done in " << (optimiz - inter2) / (double)(CLOCKS_PER_SEC / 1000000) << " s. Flow: " << std::to_string(flow) << std::endl;
        std::cout << "    Optimization done in " << (optimiz - inter2) / (double)(CLOCKS_PER_SEC) << " s.\n    Flow: " << std::to_string(flow) << std::endl << std::endl;
        
        bool flag = true;
        
        for (int i = 0; i < n_nodes; i++){
            if (g->what_segment(i) == GraphType::SOURCE){
                flag = false;
                //std::cout << std::to_string(i) << "Quitting\n";
                }
            }
        
        
        
        
        
        if (flag){
            std::cout << "Empty solution found. Restarting optimization with translated graph.\n\n";
            repeat = true;
            if (count > 0)
                break;
            count++;
            float sum = 1.;
            for (int x=0; x < shape_weights[0]; ++x){
                for (int y=0; y < shape_weights[1]; ++y){
                    for (int z=0; z < shape_weights[2]; ++z){
                        sum += unaries_cost_mat[weights[x][y][z]][0];
                    }
                }
            }
            
            offset = sum;
            printf("\nOFFSET: %f\n", offset);
            
            second_iteration = true;
            
    
        }
        else{
            repeat = false;

                    

            std::cout << "Writing results to file." << std::endl;
            std::ofstream myfile (output);
            if (myfile.is_open())
              {
                myfile << std::to_string(g->what_segment(0) == GraphType::SOURCE) ;
                for(int i = 1; i < n_nodes; i++){
                    myfile << " " << std::to_string(g->what_segment(i) == GraphType::SOURCE) ;
                }
                myfile.close();
              }

            }
            
        if (second_iteration)
            delete g;
    }
    
    infile.close();
	return 0;
}
































int graph_cut2D(std::string input, std::string output)
{
        
    
    
    std::clock_t    start, init, inter1, inter2, graph, optimiz;
    start = std::clock();
    int imax = std::numeric_limits<int>::max();
    float fmax = std::numeric_limits<float>::max();
    
    float INF = fmax;
    


    std::ifstream infile(input);
    std::string line;
    
    
    printf("Starting...");
    // 1st non-commented line is the number of nodes.
    int n_nodes, max_dist, dims;
    
    while(std::getline(infile, line)){
        printf("%c\n",line[0]);
        if (line[0] != '#')
            break;
    }
    
    std::istringstream iss(line);
    iss >> n_nodes >> max_dist >> dims;
    int* shape_weights = new int[dims];
    for (int i=0; i<dims; i++)
        iss >> shape_weights[i];
        
    printf("N_NODES: %d, %d, %d, dimensions: %d,%d,%d\n",n_nodes, max_dist, dims, shape_weights[0], shape_weights[1], shape_weights[2]);
    
    
    
    int* shape_neigh = new int[dims];
    for (int i=0; i<dims; i++)
        iss >> shape_neigh[i];
    
    iss.clear();//clear any bits set
    iss.str(std::string());

    
    
    
    
    
    
    
    // 2nd non-commented line is the neighborhood-structure data.
    int*** neighborhood = new int**[shape_neigh[0]];
    for(int i = 0; i < shape_neigh[0]; ++i){
        neighborhood[i] = new int*[shape_neigh[1]];
        for(int j = 0; j < shape_neigh[1]; ++j){
                neighborhood[i][j] = new int[shape_neigh[2]];
        }
    }
    while(std::getline(infile, line)){
        if (line[0] != '#')
            break;
    }
    iss.clear();//clear any bits set
    iss.str(std::string());
    iss.str(line);
    int n_neighbors = 0;
    for(int i = 0; i < shape_neigh[0]; ++i)
        for(int j = 0; j < shape_neigh[1]; ++j)
            for(int k = 0; k < shape_neigh[2]; ++k){
                iss >> neighborhood[i][j][k];
                n_neighbors += neighborhood[i][j][k];
                }  
    
    
    
    
    
    
    // 3rd non-commented line is the weights data.
    float*** weights = new float**[shape_weights[0]];
    for(int i = 0; i < shape_weights[0]; ++i){
        weights[i] = new float*[shape_weights[1]];
        for(int j = 0; j < shape_weights[1]; ++j){
            weights[i][j] = new float[shape_weights[2]];
        }
    }
    
    while(std::getline(infile, line)){
        if (line[0] != '#')
            break;
    }
    

    iss.clear();//clear any bits set
    iss.str(std::string());
    iss.str(line);
    for(int i = 0; i < shape_weights[0]; ++i)
        for(int j = 0; j < shape_weights[1]; ++j)
            for(int k = 0; k < shape_weights[2]; ++k){
                float my_number;
                iss >> my_number;//weights[i][j][k];
                weights[i][j][k] = my_number;
                
            }
                    
    
    
    
    // BUILD GRAPH WITH NODES
    bool repeat = true;
    int count = 0;
    bool second_iteration = false;
    
    init = std::clock();
    while (repeat){
        inter1 = std::clock();
        
        typedef Graph<float,float,float> GraphType;

        GraphType *g = new GraphType(/*estimated # of nodes*/ n_nodes, /*estimated # of edges*/ n_neighbors*n_nodes); 
        
        for (int i=0; i<n_nodes; ++i)
            g -> add_node();
            
        
        
        
        
        
        // ADD INTERNAL EDGES
        
        int yz = shape_weights[1]*shape_weights[2];
        int z_row = shape_weights[2];
        
        int ind;
        int* half_shape = new int[dims];
        int edge_count = 0;
        int tedge_count = 0;
        half_shape[0] = (shape_neigh[0]-1)/2;
        half_shape[1] = (shape_neigh[1]-1)/2;
        half_shape[2] = (shape_neigh[2]-1)/2;
        
        for (int x=0; x < shape_weights[0]; ++x){
            for (int y=0; y < shape_weights[1]; ++y){
                for (int z=0; z < shape_weights[2]; ++z){
                    
                        int ind_src = yz*x + z_row*y + z;
                        
                        for (int sx=0; sx < shape_neigh[0]; ++sx){
                            for (int sy=0; sy < shape_neigh[1]; ++sy){
                                for (int sz=0; sz < shape_neigh[2]; ++sz){
                                    
                                        int px,py,pz;
                                        px = sx-half_shape[0];
                                        py = sy-half_shape[1];
                                        pz = sz-half_shape[2];
                                        
                                        
                                        //printf ("%d, %d, %d\n", sx, sy, sz);;
                                        if (neighborhood[sx][sy][sz] != 0){
                                        
                                            if (x+px >= 0 && x+px < shape_weights[0] && y+py >= 0 && y+py < shape_weights[1] && z+pz >= 0 && z+pz < shape_weights[2]){
                                                
                                                int ind_dest = (x+px)*yz + (y+py)*z_row + (z+pz);
                                                
                                                //printf ("%d, %d, %d, %d  ->  ", x, y, z, w);
                                                //printf ("%d, %d, %d, %d\n", x+px, y+py, z+pz, w+pw);
                                                //printf ("%d -> %d\n", ind_src, ind_dest);
                                                
                                                g -> add_edge( ind_src, ind_dest,    /* capacities */  INF, 0. );
                                                edge_count++;
                                            }
                                            else
                                                if (x+px >= 0 && x+px < shape_weights[0] && y+py >= 0 && y+py < shape_weights[1] && z+pz  < 0){
                                                    
                                                    int ind_dest = (x+px)*yz + (y+py)*z_row + 0;
                                                    
                                                    //printf ("%d -> %d\n", ind_src, ind_dest);
                                                    if (ind_src != ind_dest){
                                                        //printf ("%d, %d, %d, %d  ->  ", x, y, z, w);
                                                        //printf ("%d, %d, %d, %d\n", x+px, y+py, z+pz, w+pw);
                                                        g -> add_edge( ind_src, ind_dest,    /* capacities */  INF, 0. );
                                                        edge_count++;
                                                    }
                                                }
                                            }
                                    }
                                }
                            }
                        
                                        
                        
                        if (weights[x][y][z] <= 0){
                            g -> add_tweights( ind_src,    /* capacities */   -weights[x][y][z], 0.);
                            tedge_count++;}
                        else{
                            g -> add_tweights( ind_src,    /* capacities */   0., weights[x][y][z]);
                            tedge_count++;}
                    
                }
            }
        }
        
        printf("    Number of internal edges: %d\n    Number of terminal edges: %d\n\nStarting optimization.\n", edge_count, tedge_count);
        
        graph = std::clock();
        inter2 = std::clock();
        int flow;
        flow = g -> maxflow();
        //std::cout << "\nFlow:" << std::to_string(flow) << "\n";
        optimiz = std::clock();
        
//        std::cout << "Optimization done in " << (optimiz - inter2) / (double)(CLOCKS_PER_SEC / 1000000) << " s. Flow: " << std::to_string(flow) << std::endl;
        std::cout << "Optimization done in " << (optimiz - inter2) / (double)(CLOCKS_PER_SEC) << " s. Flow: " << std::to_string(flow) << std::endl;
        
        bool flag = true;
        
        for (int i = 0; i < n_nodes; i++){
            if (g->what_segment(i) == GraphType::SOURCE){
                flag = false;
                //std::cout << std::to_string(i) << "Quitting\n";
                }
            }
        
        
        
        
        
        if (flag){
            std::cout << "Empty solution found. Translating graph.\n";
            repeat = true;
            if (count > 0)
                break;
            count++;
            float sum = 1.;
            for (int x=0; x < shape_weights[0]; ++x){
                for (int y=0; y < shape_weights[1]; ++y){
                    sum += weights[x][y][0];
                }
            }
            
            weights[0][0][0] -= sum;
            second_iteration = true;
            
    
        }
        else{
            repeat = false;

                    

            std::cout << "Writing results to file." << std::endl;
            std::ofstream myfile (output);
            if (myfile.is_open())
              {
                for(int i = 1; i < n_nodes; i++){
                }
                myfile.close();
              }

            }
            
        if (second_iteration)
            delete g;
    }
    
    infile.close();
	return 0;
}



int main(int argc, char* argv[])
{
    // ------------------------------------------------------------------------
    // 0. Parse input
    // ------------------------------------------------------------------------
    if (argc != 3) {
        // Tell the user how to run the program
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
        /* "Usage messages" are a conventional way of telling the user
         * how to run a program if they enter the command incorrectly.
         */
        return 1;
    }
    
    std::string input, output;
    input = argv[1];
    output = argv[2];
        




    // ------------------------------------------------------------------------
    // 1. Read graph from file
    // ------------------------------------------------------------------------

    // File has the following structure:
    //
    //  0. Any line starting with '#' is ignored.
    //
    //  1. First line contains space-separated integers, in order:
    //      a. the number of nodes (1 int)
    //      b. the maximal distance (1 int)
    //      c. the dimensions of the graph (call it d) (1 int)
    //      d. the shape of the graph (d ints)
    //      e. the shape of the neighborhood structure (d ints)
    //
    //  2. Then a line of space-separated integers that represents the
    //     neighborhood structure. This structure is a d-D array with odd
    //     dimensions where non-zero entries represent existing edges.
    //
    //  3. Lastly a line of space-separated floats that represents the unaries
    //     of the graph.
    //
    
    
    std::ifstream infile(input);
    std::string line;
    
    // 1st non-commented line is the number of nodes.
    int n_nodes, max_dist, dims;
    
    while(std::getline(infile, line)){
        if (line[0] != '#')
            break;
    }
    
    std::istringstream iss(line);
    iss >> n_nodes >> max_dist >> dims;
    
    infile.close();
    
    switch (dims)
    {
        case 4:
            return graph_cut3D_VCE(input, output);
            break;
        case 3:
            return graph_cut3D_VCE(input, output);
            break;
        default:
            return 1;
    }
}
