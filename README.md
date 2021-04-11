# mlp
primitive multi-layer perceptron implementation

Usage: mlp input-data output-dir epoch-max speed layers-cfg

input-data - path to text file with data: each line contains point coordinates and its class label

output-dir - path to dir with train results (will contain errs.txt, errs_val.txt, weights.txt, results.txt)

epoch-max - [int] maximum epoch number

speed - [double] learning speed

layers-cfg - path to text file containing a single line with hidden layers' outputs number




err_graph.py - script for training and validation errors visualisation

Usage: python err_graph.py <output-dir>




draw_results.py - script for classification results visualisation

Usage: python draw_results.py results-path
results-path: output-dir/results.txt
