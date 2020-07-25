# BenchmarkIDS

Java CLI application to benchmark IDS, using WEKA's Java API and CSE-CIC-IDS2018 dataset.

This is the supporting application of my MSc dissertation on benchmarking behavior-based IDS using bio-inspired algorithms.

It's a very specific application, mainly aimed at automating the specific tests used on my dissertation.

Still, it´s a nice example on how to use WEKA's Java API.

The BenchmarkIDS application was created as a result of meeting some needs related to this work, namely the flexibility provided by the API and trying to overcome some limitations of the graphic environment. The application, developed in Java, has no graphical environment and the results are only shown on the screen (they are not saved in a file).

## Structure
The application consists of 4 modules: the initialization module (\ textit {setup}), the training module, the testing module and the scenarios module.

**The initialization module** is responsible for creating the data files that will be used by the training and testing modules.

Starting from the parameters passed on the command line, open the corresponding files from the dataset, do the pre-processing and generate the files for training the model (randomly selects 200000 lines from the file and divides them into two files, one with 140,000 lines (70 \%) for training and the other with the remaining 60,000 (30 \%), used to test the model) and for zero-day attack simulation testing.

**The training module** reads the previously generated data file and trains the model, using predefined algorithms. The parameterization of the algorithms is also pre-defined, with 3 models of each algorithm being generated.

**The test module** loads the previously generated models, reads the test file and the Zero-Day attack simulation file and tests and evaluates the models, generating a set of metrics indicative of the performance of each model.

**The scenarios module**, which implements the 4 test scenarios of this work, loads the files previously generated from the CSE-CIC-IDS2018 dataset by the Orange application, selects from one of them 140000 lines for learning and 60,000 lines of the other that is intended for the test phase and applies this data to the 3 algorithms used. For reasons of statistical significance, this procedure is repeated ten times, with the data selected each time being completely independent of the others.

## Compiling

## Usage

To use this application it is necessary to open the Windows command line (or, if using Linux, a shell or terminal) and change to the folder (directory) where the application distribution is located.

### Initialization module
In order to execute the initialization module and thus generate the necessary data files, a command with the following structure must be executed:

java -jar BenchmarkIDS.jar setup <fich1> <fich2>



The parameters mean the following:
\ begin {itemize}
\ item <fich1> - file number that will serve as the basis for training the model
\ item <fich2> - file number that will serve as the basis for the zero-day attack simulation
\ end {itemize}

So, for example, the command

\ begin {Verbatim}
java -jar BenchmarkIDS.jar setup 02 03
\ end {Verbatim}

will execute the initialization module, accessing the Data \ _02.csv file to generate the model training data and the Data \ _03.csv file to create the file that will simulate the zero-day attack.

In the training module, the command structure is as follows:

\ begin {Verbatim *}
java -jar BenchmarkIDS.jar training <algoritmo>
\ end {Verbatim *}

The algorithm parameter can only assume 3 distinct values:

\ begin {itemize}
\ item clonalg - to select the CLONALG algorithm
\ item mlp - to select the Multi-Layer Perceptron algorithm
\ item lvq - to select the Learning Vector Quantization algorithm
\ end {itemize}

If, for example, we intended to train CLONALG models, the command to be executed would be, then:

\ begin {Verbatim}
java -jar BenchmarkIDS.jar training clonalg
\ end {Verbatim}

The test module includes two different options, one to test a certain algorithm and the other to generate and test the \ textit {ensembles} of the 3 algorithms, grouped 2 to 2.

The command structure for testing a given algorithm, including the zero-day attack simulation test, is as follows:

\ begin {Verbatim *}
java -jar BenchmarkIDS.jar test <algoritmo>
\ end {Verbatim *}

The admissible values ​​for the parameter <algorithm> are the same as for the parameter with the same name in the training module.

To generate the \ textit {ensembles} and perform the respective tests, simply execute the command:

\ begin {Verbatim *}
java -jar BenchmarkIDS.jar ensemble
\ end {Verbatim *}

To use the scenarios module, simply execute the command:

\ begin {Verbatim *}
java -jar BenchmarkIDS.jar training <cenario>
\ end {Verbatim *}

The parameter \ textit {scenario} can take the values ​​scenario1, scenario2, scenario3 and scenario4.


