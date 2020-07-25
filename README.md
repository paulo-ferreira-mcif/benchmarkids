# BenchmarkIDS

Java CLI application to benchmark IDS, using WEKA's Java API and CSE-CIC-IDS2018 dataset.

This is the supporting application of my MSc dissertation on benchmarking behavior-based IDS using bio-inspired algorithms.

It's a very specific application, mainly aimed at automating the specific tests used on my dissertation.

Still, it´s a nice example on how to use WEKA's Java API.

The BenchmarkIDS application was created as a result of meeting some needs related to this work, namely the flexibility provided by the API and trying to overcome some limitations of the graphic environment. The application, developed in Java, has no graphical environment and the results are only shown on the screen (they are not saved in a file).

## {Structure}
The application consists of 4 modules: the initialization module (\ textit {setup}), the training module, the testing module and the scenarios module.

**The initialization module** is responsible for creating the data files that will be used by the training and testing modules.

Starting from the parameters passed on the command line, open the corresponding files from the dataset, do the pre-processing and generate the files for training the model (randomly selects 200000 lines from the file and divides them into two files, one with 140,000 lines (70 \%) for training and the other with the remaining 60,000 (30 \%), used to test the model) and for zero-day attack simulation testing.

**The training module** reads the previously generated data file and trains the model, using predefined algorithms. The parameterization of the algorithms is also pre-defined, with 3 models of each algorithm being generated.

**The test module** loads the previously generated models, reads the test file and the Zero-Day attack simulation file and tests and evaluates the models, generating a set of metrics indicative of the performance of each model.

**The scenarios module**, which implements the 4 test scenarios of this work, loads the files previously generated from the CSE-CIC-IDS2018 dataset by the Orange application, selects from one of them 140000 lines for learning and 60,000 lines of the other that is intended for the test phase and applies this data to the 3 algorithms used. For reasons of statistical significance, this procedure is repeated ten times, with the data selected each time being completely independent of the others.


