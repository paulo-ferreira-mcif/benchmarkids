# BenchmarkIDS

Java CLI application to benchmark IDS, using WEKA's Java API and CSE-CIC-IDS2018 dataset.

This is the supporting application of my MSc dissertation on "Benchmarking behavior-based IDS using bio-inspired algorithms".

It's a very specific application, mainly aimed at automating the specific tests used on my dissertation.

Still, it´s a nice example on how to use WEKA's Java API.

The BenchmarkIDS application was created as a result of meeting some requirements related to this work, namely the flexibility provided by the API and trying to overcome some limitations of the graphic environment. The application, developed in Java, has no graphical environment and the results are only shown on the screen (they are not saved in a file).

## Structure
The application consists of 4 modules: the initialization module (_setup_), the training module, the testing module and the scenarios module.

**The initialization module** is responsible for creating the data files that will be used by the training and testing modules.

Starting from the parameters passed on the command line, open the corresponding files from the dataset, performs the pre-processing tasks and generate the files for training the model (randomly selects 200000 lines from the file and divides them into two files, one with 140,000 lines (70%) for training and the other with the remaining 60,000 (30%), used to test the model) and for zero-day attack simulation testing.

**The training module** reads the previously generated data file and trains the model, using predefined algorithms. The parameterization of the algorithms is also pre-defined, with 3 models of each algorithm being generated.

**The test module** loads the previously generated models, reads the test file and the Zero-Day attack simulation file and tests and evaluates the models, generating a set of metrics indicative of the performance of each model.

**The scenarios module**, which implements the 4 test scenarios of my work, loads the files previously generated from the CSE-CIC-IDS2018 dataset by the Orange data-mining application, selects from one of them 140000 lines for the training phase and 60,000 lines of the second one for the test phase and applies this data to the 3 algorithms used. For reasons of statistical significance, this procedure is repeated ten times, with the data selected each time being completely independent of the others.

## Compiling

I compiled the code using Netbeans version 11.2

It's absolutely necessary, for the compilation to succeed:

* Include the Weka API, by means of including the file Weka.jar from your Weka installation directory
* Include the Wekaclassalgos library. This library, originally developed by Jason Brownlee, is available in github at https://github.com/fracpete/wekaclassalgos

For those using Maven, there's also a Maven dependency that deals with including Wekaclassalgos library (see https://github.com/fracpete/wekaclassalgos for more information).

Information on how to use the WEKA API in Java applications can be found at [Use weka in your java code](https://waikato.github.io/weka-wiki/use_weka_in_your_java_code/#links)

## Usage

To use this application it is necessary to open the Windows command line (or, if using Linux, a shell or terminal) and change to the folder (directory) where the application distribution is located.

### Initialization module
In order to execute the initialization module and thus generate the necessary data files, a command with the following structure must be executed:

```java -jar BenchmarkIDS.jar setup <fich1> <fich2>```



The parameters mean the following:

* <fich1> - file number that will serve as the basis for training the model
* <fich2> - file number that will serve as the basis for the zero-day attack simulation

So, for example, the command

```java -jar BenchmarkIDS.jar setup 02 03```

will execute the initialization module, accessing the Data_02.csv file to generate the model training data and the Data_03.csv file to create the file that will simulate the zero-day attack.

### Training module
In the training module, the command structure is as follows:

```java -jar BenchmarkIDS.jar training <algoritmo> ```

The **algoritmo** parameter can only assume 3 distinct values:

* clonalg - to select the CLONALG algorithm
* mlp - to select the Multi-Layer Perceptron algorithm
* lvq - to select the Learning Vector Quantization algorithm

If, for example, we intended to train CLONALG models, the command to be executed would be, then:

```java -jar BenchmarkIDS.jar training clonalg```

### Test Module
The test module includes two different options, one to test a certain algorithm and the other to generate and test the *ensembles* of the 3 algorithms, grouped 2 to 2.

The command structure for **testing a given algorithm**, including the zero-day attack simulation test, is as follows:

```java -jar BenchmarkIDS.jar test <algoritmo>```

The valid and supported values for the parameter **algoritmo** are the same as for the parameter with the same name in the training module.

To **generate the _ensembles_ and perform the respective tests**, simply execute the command:

```java -jar BenchmarkIDS.jar ensemble```

### Scenarios Module
To use the scenarios module, simply issue the command:

```java -jar BenchmarkIDS.jar training <cenario>```

The valid values for the parameter **cenario** are : 

* cenario1 - will perform the defined tests for scenario 1
* cenario2 - will perform the defined tests for scenario 2
* cenario3 - will perform the defined tests for scenario 3
* cenario4 - will perform the defined tests for scenario 4

## Notes
As mentioned before, this application was developed to meet specific requirements of my work. As such, some things were just simplified, like:

- The location of the data files is hardcoded
- The algorithms used are just the ones I used on my work
- The patametrization of each algorithm is also hardcoded
- The name of the data files resembles what I used on my work and is also hardcoded

## About the authors
Paulo Ferreira (2180047@my.ipleiria.pt) - I'm a MSc student - Cybersecurity and Computer Forensics - at Polytechnic Institute of Leiria (IPL), School of Technology and Management (ESTG)

Mário Antunes (mario.antunes@ipleiria.pt) - professor and Coordinator of the Master in Cybersecurity and Computer Forensics at Polytechnic Institute of Leiria (IPL), School of Technology and Management (ESTG), Department of Computer Engineering (DEI).
