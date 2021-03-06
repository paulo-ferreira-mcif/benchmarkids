/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package pjgf;

import java.awt.BorderLayout;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Scanner;
import java.util.logging.Level;
//import java.lang.Exception;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.immune.clonalg.CLONALG;
import weka.classifiers.meta.Vote;
import static weka.classifiers.neural.common.learning.LearningKernelFactory.LEARNING_FUNCTION_STATIC;
import static weka.classifiers.neural.common.learning.LearningKernelFactory.TAGS_LEARNING_FUNCTION;
import static weka.classifiers.neural.common.training.TrainerFactory.TAGS_TRAINING_MODE;
import static weka.classifiers.neural.common.training.TrainerFactory.TRAINER_BATCH;
import static weka.classifiers.neural.common.transfer.TransferFunctionFactory.TAGS_TRANSFER_FUNCTION;
import static weka.classifiers.neural.common.transfer.TransferFunctionFactory.TRANSFER_SIGMOID;
import weka.classifiers.neural.lvq.Lvq3;
import weka.classifiers.neural.multilayerperceptron.BackPropagation;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Normalize;
// import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

/**
 *
 * @author Paulo User
 */
public class BenchmarkIDS {
    
    // Constants
    private static final String data_path="c:\\Developer\\dados_mcif\\";
    private static final String modelos_path=data_path+"modelos\\";
    private static final String reports_path=data_path+"reports\\";
    private static final String dataset_path=data_path+"dataset\\";
    
    /* 
    * folder with dataset files to use in simulation
    * there will be 3 files: training.csv, test.csv and zeroday.csv
    */
    private static final String currentdataset_path=data_path+"current\\"; 
    
    private static final int numModelos=3;
    
    // file containing training and test data to generate the models
    private static final String ficheiro1=data_path+"Dados_02.csv";
    
    // ficheiro1_training - data from file 1; model training data
    private static final String ficheiro1_training=currentdataset_path+"training.csv";
    
    // ficheiro1_test - data from file 1; model test data
    private static final String ficheiro1_test=currentdataset_path+"test.csv";
        
    // File with test data
    private static final String ficheiro2=data_path+"Dados_03.csv";
    
    // zero-day simulation file
    private static final String zeroday_file=currentdataset_path+"zeroday.csv";
    
    /**
     * Function to open a dataset from a file
     * @param filename
     * @return 
     * 
     */
    public static Instances abreDataset(String filename) {
       
        Instances dataset;
        
        CSVLoader loader=new CSVLoader();
        
        File dados=new File(filename);
        
        // Attributes with "Infinity" as value are considered Missing Values
        loader.setMissingValue("Infinity");
        
            
        try {    
            //DataSource ficheiro=new DataSource(filename);
            //dataset=ficheiro.getDataSet();
            
            loader.setSource(dados);
            
            dataset=loader.getDataSet();
            
            // Set the Label/class column
            if (dataset.classIndex() == -1) {
                dataset.setClassIndex(dataset.numAttributes() - 1);
            }
            
            return dataset;
        } catch (Exception ex) {
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);   
            return null;
        }
    }
    
    
    // Gets number of occurrences for each distinct class value
    // The dataset must have a defined class
    public static void estatisticaClasse (Instances dados){        
        
        // Numero de valores diferentes da classe
        int numLinhas;
        int indexValor;
        String valor;
        
        Attribute classe;
                                
        classe=dados.classAttribute();
                
        Enumeration valoresClasse=classe.enumerateValues();
        
        // Itera sobre as classes
        while (valoresClasse.hasMoreElements()){            
            
            valor=valoresClasse.nextElement().toString();
            
            // numero do indice
            indexValor=classe.indexOfValue(valor);
            numLinhas=dados.attributeStats(dados.classIndex()).nominalCounts[indexValor];
            System.out.println(valor + " => " + numLinhas);
        }        
                        
    }
    
    /**
     * Function to get approximately numLinhas samples from a dataset; 
     * The result data is stratified
     * @param dados - source dataset
     * @param numLinhas - Number of lines
     * @param seed - seed to use
     * 
     * @return 
     */
    public static Instances divideDataset(Instances dados,int numLinhas,int seed){
        
        Instances result;
        int numTotalLinhas;
        double percent;

        // numTotalLinhas=dados.attributeStats(dados.classIndex()).totalCount;
        numTotalLinhas=dados.numInstances();
        System.out.println("Numero total de linhas:" + numTotalLinhas);

        percent=((numLinhas*1.0)/numTotalLinhas)*100;

        System.out.println("Percentagem:" + percent);

        Resample filtro=new Resample();

        // 0.0 => distribuição da classe como está
        // 1.0 => classe com distribuição uniformizada
        filtro.setBiasToUniformClass(0.0);

        // Sem reposição das instâncias seleccionadas
        filtro.setNoReplacement(true);

        // Percentagem a amostrar
        filtro.setSampleSizePercent(percent);

        // Seed
        filtro.setSeed(seed);
            
        try {    
            filtro.setInputFormat(dados);       // prepare the filter for the data format
            
            filtro.setInvertSelection(false);  // do not invert the selection
            
            // apply filter
            result = Filter.useFilter(dados, filtro);
                            
            return result;
        } catch (Exception ex) {
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);
            return null;
        }
    }
    
    
    // Function to change class values; final values will be:
    // => Normal
    // => Malicioso
    public static Instances trataClasse(Instances dados){
        
        int numLinhas;        
        int indiceClasse,indiceTrafego,indiceTimestamp;        
        Attribute classe;
        List valores;
        
        numLinhas=dados.numInstances();
        
        indiceClasse=dados.classIndex();
        
        classe=dados.classAttribute();
        
        valores=new ArrayList();
        
        // Valores possíveis para o novo atributo
        valores.add("Normal");
        valores.add("Malicioso");
        
        // Cria o atributo
        dados.insertAttributeAt(new Attribute("Trafego",valores), dados.numAttributes());
        
        // Indice do atributo Trafego
        indiceTrafego=dados.attribute("Trafego").index();
        
        // Preenche a coluna do novo atributo com base no valor de Label
        for (int i=0; i<numLinhas;i++){
            // System.out.print("NumLinha: " + i + " => ");
            // System.out.print(dados.instance(i).toString(indiceClasse));
            if (dados.instance(i).toString(indiceClasse).equalsIgnoreCase("Benign")){                
                dados.instance(i).setValue(indiceTrafego, "Normal");
                // System.out.println(" => Normal");
            } else {              
                dados.instance(i).setValue(indiceTrafego, "Malicioso");
                // System.out.println(" => Malicioso");
            }
        }
        
        // Altera o atributo de classificação
        dados.setClassIndex(indiceTrafego);
        
        // Remove atributo Label
        dados.deleteAttributeAt(indiceClasse);
        
        // Remove o timestamp, pois o BackPropagationMLP
        // não consegue tatar dois campos nominal...
        indiceTimestamp=dados.attribute("Timestamp").index();
        dados.deleteAttributeAt(indiceTimestamp);
        
        return dados;
    }
    
    
    /**
     * Function to deal with missing values in a specified dataset
     * 
     * @param dados - dataset to use
     * 
     * @return 
     *  
     */
    public static Instances trataMissingValues(Instances dados) {
        Instances result;
        
        
        ReplaceMissingValues filter=new ReplaceMissingValues();
        //RemoveWithValues filter=new RemoveWithValues();
        
        //filter.setAttributeIndex(Integer.toString(field));
        //filter.setMatchMissingValues(true);
        try{
            filter.setInputFormat(dados);
            result=Filter.useFilter(dados, filter);
            return result;                           
        } catch (Exception ex) {
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);
            return null;
        }
    }
    
    
    /**
     * Generate a model using CLONALG algorithm
     * @param dadosTreino : dataset used to train the model
     * @param options : model parameters
     * @return Returns a CLONALG model with specified configuration
     */
    public static Classifier geraModeloCLONALG(Instances dadosTreino,String[] options){
        // Classifier modelo;
        
        CLONALG clonalg;
        
        clonalg=new CLONALG();                
                        
        try {
            clonalg.setOptions(options);
            clonalg.buildClassifier(dadosTreino);
        } catch (Exception ex) {
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(0);
        }
        
        return clonalg;
    }
    
    /**
     * Function to generate an array of strings containing configuration options
     * The array will be used in creating models using the Jason Brownlee implementation of BackMLP algorithm
     * Parameters correspond to Jason Brownlee's implementation
     * @param layer1
     * @param layer2
     * @param layer3
     * @param bias
     * @param learningRate
     * @param learningRateFunction
     * @param momentum
     * @param iterations
     * @param transfer
     * @param weightDecay
     * @param seed
     * @return 
     */
    public static String[] geraOptBackMLPBrownlee(int layer1,int layer2,int layer3,
            double bias,double learningRate,int learningRateFunction,double momentum,
            int iterations,int transfer,double weightDecay,int seed){
        
        String[] opt=new String[22];
        
        opt[0]="-X"; // hidden layer 1
        opt[1]=Integer.toString(layer1);
        opt[2]="-Y"; // hidden layer 2
        opt[3]=Integer.toString(layer2);
        opt[4]="-Z"; // hidden layer 3
        opt[5]=Integer.toString(layer3);
        opt[6]="-B"; // Bias Input
        opt[7]=Double.toString(bias);
        opt[8]="-L"; // Learning Rate
        opt[9]=Double.toString(learningRate);
        opt[10]="-M"; // Learning Rate Function
        opt[11]=Integer.toString(learningRateFunction);
        opt[12]="-I"; // Training iterations
        opt[13]=Integer.toString(iterations);
        opt[14]="-A"; // Momentum
        opt[15]=Double.toString(momentum);
        opt[16]="-F"; // Transfer Function
        opt[17]=Integer.toString(transfer);
        opt[18]="-D"; // Weight Decay
        opt[19]=Double.toString(weightDecay);
        opt[20]="-R"; // Seed
        opt[21]=Integer.toString(seed);
                                        
        return opt;
    }
    
    
    /**
     * 
     * Function to generate an array of strings containing configuration options
     * The array will be used in creating models using BackMLP algorithm (WEKA implementation)
     * @param hiddenLayers
     * @param learningRate
     * @param momentum
     * @param trainingTime
     * @param seed
     * @return 
     */
    public static String[] geraOptBackMLP(String hiddenLayers,double learningRate,
            double momentum, int trainingTime,int seed){
        
        String[] opt=new String[17];
        
        opt[0]="-H"; // hidden layers
        opt[1]=hiddenLayers;
        opt[2]="-L"; // Learning Rate
        opt[3]=Double.toString(learningRate);
        opt[4]="-M"; // Momentum
        opt[5]=Double.toString(momentum);
        opt[6]="-N"; // Training Time (epochs)
        opt[7]=Integer.toString(trainingTime);
        opt[8]="-V"; // Validation Set Size (0=> will train for the specified epochs)
        opt[9]=Integer.toString(0);
        opt[10]="-E"; // Validation Threshold
        opt[11]=Integer.toString(20); // default value
        opt[12]="-B"; // Disable BinaryToNominalFilter
        opt[13]="-I"; // Disable Normalize filter; data will be normalized outside
        opt[14]="-C"; // Do not normalize class
        opt[15]="-S"; // Seed
        opt[16]=Integer.toString(seed);
                                
        return opt;
    }
    
    /**
     * Function to generate a model using Jason Brownlee's implementation of BackMLP
     * 
     * @param dadosTreino - Training data (MUST BE NORMALIZED!)
     * @param options - model configuration options
     * @return 
     *  
     */
     public static Classifier geraModeloBackMLPBrownlee(Instances dadosTreino,String [] options) {
        BackPropagation bpmlp;                  
        
        bpmlp=new BackPropagation();
        
        try{        
            bpmlp.setOptions(options);
            bpmlp.buildClassifier(dadosTreino);
        } catch (Exception ex){
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(0);
        }
                 
        return bpmlp;               
    }
     
    
    /**
     * Function to generate a model using WEKA's implementation of BackMLP
     * 
     * @param dadosTreino - Training data (MUST BE NORMALIZED!)
     * @param options - model configuration options
     * @return 
     *  
     */
     public static Classifier geraModeloBackMLP(Instances dadosTreino,String [] options) { 
         MultilayerPerceptron bpmlp;                  
        
        bpmlp=new MultilayerPerceptron();
        
        try{        
            bpmlp.setOptions(options);
            bpmlp.buildClassifier(dadosTreino);
        } catch (Exception ex){
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(0);
        }
                 
        return bpmlp;
     }
     
    /**
     * Function to generate the 3 BackMLP (Jason Brownlee's implementation) models used in my thesis
     * The Brownlee implementation was not used in the final work, but the code was maintained here as reference
     * The models will be saved, for later use, in the pre-defined folder
     * @param dadosTreino
     * @param seed 
     */
    static void geraModelosBackMLPBrownlee(Instances dadosTreino, int seed){
        Classifier modelo;
        
        String [] backMLP1,backMLP2,backMLP3;
        
        // Geração dos modelos BackMLP
        /* Opções:
        
        biasInput: recommended 1.0
        hiddenLayer1: number of nodes in layer 1 (0 for none)
        hiddenLayer2: number of nodes in layer 2 (0 for none)
        hiddenLayer3: number of nodes in layer 3 (0 for none)
        learningRate: Learning Rate - between 0.05 and 0.75 (recommended 0.1)
        learningRateFunction: 1=linear decay; 2=Inverse;3=static
        momentum: momentum factor; between 0.0 and 0.9; 0.0=not used
        randomNumberSeed: seed
        trainingIterations: number of iterations; between few hundred and few thousands
        transferFunction: neuron transfer function
            1: sigmoid
            2: tanh (hyperbolic tangent)
            3: sign function (Bi-poler Step)
            4: Step function
            5:  Gaussian function
        weightDecay: weight decay factor; between 0.0 and 1.0 (0.0=not used)
        */
        
        /* Opções do modelo BackMLP1
            layer1: 80
            layer2: 40
            layer3: 10
            biasInput: 1.0
            learningRate: 0.1
            learningRateFunction: 3
            momentum: 0.0
            iterations: 500
            transferFunction: 1
            weightDecay: 0.0
        */        
        backMLP1=geraOptBackMLPBrownlee(80,40,10,1.0,0.1,3,0.0,500,1,0.0,seed);
        
        
        /* Opções do modelo BackMLP2 - altera transfer function e momentum
            layer1: 80
            layer2: 40
            layer3: 10
            biasInput: 1.0
            learningRate: 0.1
            learningRateFunction: 3
            momentum: 0.3
            iterations: 500
            transferFunction: 1
            weightDecay: 0.0
        */                
        backMLP2=geraOptBackMLPBrownlee(80,40,10,1.0,0.1,3,0.3,500,1,0.0,seed);
        
        
        /* Opções do modelo BackMLP3 - altera layer1,layer2,layer3, momentum e weightDecay
            layer1: 100
            layer2: 80
            layer3: 40
            biasInput: 1.0
            learningRate: 0.1
            learningRateFunction: 3
            momentum: 0.3
            iterations: 500
            transferFunction: 1
            weightDecay: 0.2
        */                
        backMLP3=geraOptBackMLPBrownlee(100,80,40,1.0,0.1,3,0.3,500,1,0.2,seed);
        
        
        // Conjunto das opções arrumado num array; facilita a automatização e a geração dos 3 modelos
        String[][] optBackMLP={backMLP1,backMLP2,backMLP3};
        
        // Normaliza os dados; necessário para BackMLP...
        //Normalize filtro=new Normalize();
        
        // Gera os 3 modelos, com um ciclo
        try {
            
            // Aplica o filtro            
            dadosTreino=normalizaDados(dadosTreino);
            
            for (int i=0; i<optBackMLP.length;i++){

                System.out.println("Gerando o modelo BackMLPBrownlee"+Integer.toString(i));

                modelo=geraModeloBackMLPBrownlee(dadosTreino,optBackMLP[i]);

                String nomefich1=modelos_path+"modBackMLPJB"+Integer.toString(i);

                // Guarda o modelo
                SerializationHelper.write(nomefich1,modelo);

            }
        } catch (Exception ex){
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);            
        }                                
    }
    
    /**
     * Function to generate the 3 BackMLP (WEKA implementation) models used in my thesis
     * @param dadosTreino
     * @param seed 
     */
    static void geraModelosBackMLP(Instances dadosTreino, int seed){
        Classifier modelo;
        
        String [] backMLP1,backMLP2,backMLP3;
        
        // Geração dos modelos BackMLP
        /* Opções:
        
        biasInput: recommended 1.0
        hiddenLayer1: number of nodes in layer 1 (0 for none)
        hiddenLayer2: number of nodes in layer 2 (0 for none)
        hiddenLayer3: number of nodes in layer 3 (0 for none)
        learningRate: Learning Rate - between 0.05 and 0.75 (recommended 0.1)
        learningRateFunction: 1=linear decay; 2=Inverse;3=static
        momentum: momentum factor; between 0.0 and 0.9; 0.0=not used
        randomNumberSeed: seed
        trainingIterations: number of iterations; between few hundred and few thousands
        transferFunction: neuron transfer function
            1: sigmoid
            2: tanh (hyperbolic tangent)
            3: sign function (Bi-poler Step)
            4: Step function
            5:  Gaussian function
        weightDecay: weight decay factor; between 0.0 and 1.0 (0.0=not used)
        */
        
        /* Opções do modelo BackMLP1
            hiddenLayers: 80,40,10            
            learningRate: 0.1
            momentum: 0.0
            iterations: 500
        */        
        backMLP1=geraOptBackMLP("80,40,10",0.1,0.0,500,seed);
        
        
        /* Opções do modelo BackMLP2 - altera transfer function e momentum
            hiddenLayers: 80,40,10
            learningRate: 0.1
            momentum: 0.3
            iterations: 500          
        */                
        backMLP2=geraOptBackMLP("80,40,10",0.1,0.3,500,seed);
        
        
        /* Opções do modelo BackMLP3 - altera layer1,layer2,layer3, momentum e weightDecay
            hiddenLayers: a (o mesmo é dizer (nº atributos+nº classes)/2)
            learningRate: 0.1
            momentum: 0.3
            iterations: 500
        */                
        backMLP3=geraOptBackMLP("a",0.1,0.3,500,seed);
        
        
        // Conjunto das opções arrumado num array; facilita a automatização e a geração dos 3 modelos
        String[][] optBackMLP={backMLP1,backMLP2,backMLP3};
        
        // Normaliza os dados; necessário para BackMLP...
        //Normalize filtro=new Normalize();
        
        // Gera os 3 modelos, com um ciclo
        try {
            
            // Aplica o filtro            
            // dadosTreino=normalizaDados(dadosTreino);
            
            for (int i=0; i<optBackMLP.length;i++){

                System.out.println("Gerando o modelo BackMLP"+Integer.toString(i));

                modelo=geraModeloBackMLP(dadosTreino,optBackMLP[i]);

                String nomefich1=modelos_path+"modBackMLP"+Integer.toString(i);

                // Guarda o modelo
                SerializationHelper.write(nomefich1,modelo);

            }
        } catch (Exception ex){
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);            
        }                                
    }
    
    
    /**
     * Function to test the 3 generated BackMLP models
     * @param dadosTreino
     * @param dadosTeste
     * @param seed 
     */
    public static void testaModelosBackMLP(Instances dadosTreino,Instances dadosTeste,int seed){
        Vote ensemble;             
        ArrayList predictions;  
        
        try {
            // Aplica o filtro
            //filtro.setInputFormat(dadosTreino);
            //dadosTreino=normalizaDados(dadosTreino);
                        
            //dadosTeste=normalizaDados(dadosTeste);
            
        // Le os modelos e avalia-os com os dados de teste
            for (int i=0;i<numModelos;i++){
                // Lê o modelo
                String nomefich=modelos_path+"modBackMLP"+Integer.toString(i);
                Classifier cls=(Classifier)weka.core.SerializationHelper.read(nomefich);

                // predictions... para tratamento futuro
                predictions=testaModelo(cls,dadosTreino,dadosTeste);
                
                geraROCCurve(predictions);
            }  
        } catch (Exception ex){
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);            
        }
        
        
        ensemble=geraEnsembleBackMLP(seed);                
        
        System.out.println("Testando o Classificador Ensemble");
        predictions=testaModelo(ensemble,dadosTreino,dadosTeste);
        
        geraROCCurve(predictions);
    }
    
    /**
     * Function to generate an ensemble using the 3 generated BackMLP models
     * The Majority Voting strategy was used
     * @param seed
     * @return 
     */
    public static Vote geraEnsembleBackMLP(int seed){
        Vote ensemble;
        // Majority Voting
        ensemble = new Vote();
        
        SelectedTag tag = new SelectedTag(Vote.MAJORITY_VOTING_RULE,Vote.TAGS_RULES);
        ensemble.setCombinationRule(tag);
        
        ensemble.setSeed(seed);
        
        File[] preBuiltClassifiers=new File[3];
        
        System.out.println("Gerando o Classificador Ensemble BackMLP");
        
        try {
            for (int i=0;i<numModelos;i++){
                String nome=modelos_path+"modBackMLP"+Integer.toString(i);
                Classifier cls=(Classifier)weka.core.SerializationHelper.read(nome);
                //preBuiltClassifiers[i]=new File(nome);

                System.out.println("Adicionando o modelo "+nome);

                ensemble.addPreBuiltClassifier(cls);
            }
            return (ensemble);
                                    
        } catch (Exception ex){
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);
            return (null);
        }
    }
    
    /**
     * Function to simulate a zero-day attack, using the generated BackMLP models
     * @param seed 
     */
    public static void zeroDayBackMLP(int seed){
        Vote ensemble;
        // Abre segundo ficheiro de dados e usa-o  como dataset de testes
         
        Instances dataset;
        ArrayList predictions;
        
        System.out.println("=> Abre ficheiro de dados");
        // Abre o ficheiro de treino
        dataset=abreDataset(zeroday_file);
        
        estatisticaClasse(dataset);
        
        /* System.out.println("=> Trata Classe");
        // Trata a classe
        dataset=trataClasse(dataset);
        
        estatisticaClasse(dataset);
        
        dataset=trataMissingValues(dataset); */
        
        dataset=normalizaDados(dataset);
        
        // Para efeitos de teste da ferramenta, vai buscar apenas 100000 amostras
        // No teste final, comentar as duas linhas seguintes
        // System.out.println("=> Faz o Resample");
        // dataset=divideDataset(dataset,100000,seed);
        
        try{
        // Le os modelos e avalia-os com os dados de teste
            for (int i=0;i<numModelos;i++){
                // Lê o modelo
                String nomefich=modelos_path+"modBackMLP"+Integer.toString(i);
                Classifier cls=(Classifier)weka.core.SerializationHelper.read(nomefich);

                // predictions... para tratamento futuro
                predictions=testaModelo(cls,dataset,dataset);
                
                geraROCCurve(predictions);
            }
        } catch (Exception ex) {
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);
        }
                
        ensemble=geraEnsembleBackMLP(seed);
        
        System.out.println("Testando o Classificador Ensemble");
        predictions=testaModelo(ensemble,dataset,dataset);
        
        geraROCCurve(predictions);

    }
    
    /*
     * Function to generate a LVQ (Learning Vector Quantization) model/classifier
     * using the specified options
    */
    
    public static Classifier geraModeloLVQ(Instances dadosTreino,String [] options){
        Lvq3 lvq=new Lvq3();
        
        try{        
            lvq.setOptions(options);
            lvq.buildClassifier(dadosTreino);
        } catch (Exception ex){
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(0);
        }
        
        return lvq;
    }
    
    /**
     * Function to generate the string array containing the LVQ3 model configuration options (parameters)
     * @param epsilon
     * @param initMode
     * @param learningFunction
     * @param learningRate
     * @param totalCBookVectors
     * @param iterations
     * @param useVoting
     * @param windowSize
     * @param seed
     * @return 
     */
    public static String[] geraOptLVQ3(double epsilon,int initMode,int learningFunction,
                                        double learningRate,int totalCBookVectors,
                                        int iterations,boolean useVoting,
                                        double windowSize,int seed){
        String[] opt=new String[18];
        
        opt[0]="-E"; // Epsilon
        opt[1]=Double.toString(epsilon);
        opt[2]="-M"; // Initialisation Mode
        opt[3]=Integer.toString(initMode);
        opt[4]="-L"; // Learning Function
        opt[5]=Integer.toString(learningFunction);
        opt[6]="-R"; // Learning Rate
        opt[7]=Double.toString(learningRate);
        opt[8]="-C"; // Total CodeBook Vectors
        opt[9]=Integer.toString(totalCBookVectors);
        opt[10]="-I"; // Total Iterations
        opt[11]=Integer.toString(iterations);
        opt[12]="-G"; // Use Voting?
        opt[13]=Boolean.toString(useVoting);
        opt[14]="-W"; // Window Size
        opt[15]=Double.toString(windowSize);        
        opt[16]="-S"; // Seed
        opt[17]=Integer.toString(seed);
        
        return opt;
    }
    
    /**
     * Function to generate the 3 LVQ models used in my thesis
     * @param dadosTreino
     * @param seed 
     */
    public static void geraModelosLVQ3(Instances dadosTreino,int seed){                
        
        Classifier modelo;
        
        String [] optLVQ1,optLVQ2,optLVQ3;
        
        /*
            OPÇÕES DO MODELO LVQ3
        Epsilon: learning weight modifier used when both BMUs are of the instances class (0.1 or 0.5)
        Initialisation Mode: Model (codebook vector) initalisation mode
            1==Random Training Data Proportional
            2==Random Training Data Even
            3==Random Values In Range
            4==Simple KMeans
            5==Farthest First
            6==K-Nearest Neighbour Even
        Learning Function: Learning rate function to use while training; linear is better
            1==Linear Decay
            2==Inverse
            3==Static
        Learning Rate: Initial learning rate value (recommend  0.3 or 0.5)
        Total Codebook Vectors: Total number of codebook vectors in the model (one for each attribute?)
        Total Iterations: Total number of training iterations (recommended 30 to 50 times the number of codebook vectors).
        useVoting: Use dynamic voting to select the assigned class of each codebook vector
                    provides automatic handling of misclassified instances.
        Window Size: windowSize -- Window size matching codebook vectors must be within (recommend 0.2 or 0.3)
        */
        
        /*
            Modelo 1
        
            epsilon: 0.1
            initMode: 1
            learningFunction: 1
            learningRate: 0.3
            totalCBookVectors: 90
            iterations: 4500
            useVoting: false
            windowSize: 0.2
            
        */
        optLVQ1=geraOptLVQ3(0.1,1,1,0.3,90,4500,false,0.2,seed);
        
        
        /*
            Modelo 2 - altera initMode, learningRate e useVoting
        
            epsilon: 0.1
            initMode: 6
            learningFunction: 1
            learningRate: 0.5
            totalCBookVectors: 90
            iterations: 4500
            useVoting: true
            windowSize: 0.2
        */
        optLVQ2=geraOptLVQ3(0.1,6,1,0.5,90,4500,true,0.2,seed);
        
        
        /*
            Modelo 3 - altera totalCBookVectors, iterations e useVoting
        
            epsilon: 0.1
            initMode: 1
            learningFunction: 1
            learningRate: 0.3
            totalCBookVectors: 50
            iterations: 2500
            useVoting: true
            windowSize: 0.2
        */
        optLVQ3=geraOptLVQ3(0.1,1,1,0.3,50,2500,true,0.2,seed);
        
        String[][] optLVQ={optLVQ1,optLVQ2,optLVQ3};
        
        // Gera os 3 modelos, com um ciclo
        try {
            
            // Aplica o filtro            
            //dadosTreino=normalizaDados(dadosTreino);
            
            for (int i=0; i<optLVQ.length;i++){

                System.out.println("Gerando o modelo LVQ3_"+Integer.toString(i));

                modelo=geraModeloLVQ(dadosTreino,optLVQ[i]);

                String nomefich1=modelos_path+"modLVQ"+Integer.toString(i);

                // Guarda o modelo
                SerializationHelper.write(nomefich1,modelo);

            }
        } catch (Exception ex){
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);            
        }                                
    }
    
    /**
     * Function to test the previously generated LVQ models
     * @param dadosTreino
     * @param dadosTeste
     * @param seed 
     */
    public static void testaModelosLVQ(Instances dadosTreino,Instances dadosTeste,int seed){
        Vote ensemble;             
        ArrayList predictions;  
        
        try {
            // Aplica o filtro
            //filtro.setInputFormat(dadosTreino);
            //dadosTreino=normalizaDados(dadosTreino);
                        
            //dadosTeste=normalizaDados(dadosTeste);
            
        // Le os modelos e avalia-os com os dados de teste
            for (int i=0;i<numModelos;i++){
                // Lê o modelo
                String nomefich=modelos_path+"modLVQ"+Integer.toString(i);
                Classifier cls=(Classifier)weka.core.SerializationHelper.read(nomefich);

                // predictions... para tratamento futuro
                predictions=testaModelo(cls,dadosTreino,dadosTeste);
                
                geraROCCurve(predictions);
            }  
        } catch (Exception ex){
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);            
        }
        
        
        ensemble=geraEnsembleLVQ(seed);                
        
        System.out.println("Testando o Classificador Ensemble LVQ");
        predictions=testaModelo(ensemble,dadosTreino,dadosTeste);
        
        geraROCCurve(predictions);
    }
    
    /**
     * Function to generate the ensemble of the 3 generated LVQ models
     * The Majority Voting strategy was used
     * @param seed
     * @return 
     */
    public static Vote geraEnsembleLVQ(int seed){
        Vote ensemble;
        // Majority Voting
        ensemble = new Vote();
        
        SelectedTag tag = new SelectedTag(Vote.MAJORITY_VOTING_RULE,Vote.TAGS_RULES);
        ensemble.setCombinationRule(tag);
        
        ensemble.setSeed(seed);
        
        //File[] preBuiltClassifiers=new File[3];
        
        System.out.println("Gerando o Classificador Ensemble LVQ");
        
        try {
            for (int i=0;i<numModelos;i++){
                String nome=modelos_path+"modLVQ"+Integer.toString(i);
                Classifier cls=(Classifier)weka.core.SerializationHelper.read(nome);
                //preBuiltClassifiers[i]=new File(nome);

                System.out.println("Adicionando o modelo "+nome);

                ensemble.addPreBuiltClassifier(cls);
            }
            return (ensemble);
                                    
        } catch (Exception ex){
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);
            return (null);
        }
    }
    
    /**
     * Function to simulate a zero-day attack detection using the 3 LVQ generated models
     * @param seed 
     */
    public static void zeroDayLVQ(int seed){
        Vote ensemble;
        // Abre segundo ficheiro de dados e usa-o  como dataset de testes
         
        Instances dataset;
        ArrayList predictions;
        
        System.out.println("=> Testing LVQ - Zero Day <=");
        System.out.println("=> Abre ficheiro de dados");
        // Abre o ficheiro de treino
        dataset=abreDataset(ficheiro2);
        
        estatisticaClasse(dataset);
        
        System.out.println("=> Trata Classe");
        // Trata a classe
        dataset=trataClasse(dataset);
        
        estatisticaClasse(dataset);
        
        dataset=trataMissingValues(dataset);
        
        //dataset=normalizaDados(dataset);
        
        // Para efeitos de teste da ferramenta, vai buscar apenas 100000 amostras
        // No teste final, comentar as duas linhas seguintes
        // System.out.println("=> Faz o Resample");
        // dataset=divideDataset(dataset,100000,seed);
        
        try{
        // Le os modelos e avalia-os com os dados de teste
            for (int i=0;i<numModelos;i++){
                // Lê o modelo
                String nomefich=modelos_path+"modLVQ"+Integer.toString(i);
                Classifier cls=(Classifier)weka.core.SerializationHelper.read(nomefich);

                // predictions... para tratamento futuro
                predictions=testaModelo(cls,dataset,dataset);
                
                geraROCCurve(predictions);
            }
        } catch (Exception ex) {
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);
        }
                
        ensemble=geraEnsembleLVQ(seed);
        
        System.out.println("Testando o Classificador Ensemble LVQ - Zero Day");
        predictions=testaModelo(ensemble,dataset,dataset);
        
        geraROCCurve(predictions);
    }
    
    /**
     * Function to normalize the numeric attributes of a specified dataset
     * @param dataset
     * @return 
     */
    public static Instances normalizaDados(Instances dataset){
        Normalize filtro=new Normalize();
        
        try{
            filtro.setInputFormat(dataset);
            dataset=Filter.useFilter(dataset, filtro);
        } catch (Exception ex){
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);
        }
        return dataset;
    }
     
     
    // Testa o modelo (dados de teste) e devolve predicoes
    // A ideia é devolver um Evaluation, que possa, depois, ser tratada no
    // programa principal
    // UNUSED FUNCTION... see testaModeloEvaluation function
    /**
     * Função para testar um modelo (classificador)
     * @param modelo - modelo a testar
     * @param dadosTreino - dados de treino
     * @param dadosTeste - dados de teste
     * @return  - devolve a Evaluation
     */
    public static Evaluation testaModeloEvaluation_Old(Classifier modelo,Instances dadosTreino,Instances dadosTeste) {
        
        Evaluation eval;
        ArrayList predictions;

        // predictions=new ArrayList();
        try{              
            eval=new Evaluation(dadosTreino);

            eval.evaluateModel(modelo, dadosTeste); 
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString("===> Matriz Confusão <==="));
            System.out.println("Recall: "+eval.recall(0)+"-"+eval.recall(1));
            predictions=eval.predictions();           

            return eval;
        } catch (Exception ex) {
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);
            return null;
        } 
    }
    
    
    // Function to test a specified model, using the specified test data
    // Returns predictions, as an Evaluation object, so enabling further treatment in main program
    /**
     * Função para testar um modelo (classificador), guardando os resultados no ficheiro indicado
     * @param modelo
     * @param dadosTreino
     * @param dadosTeste
     * @param file
     * @return 
     */
    public static Evaluation testaModeloEvaluation(Classifier modelo,Instances dadosTreino,Instances dadosTeste) {
        
        Evaluation eval;
        ArrayList predictions;

        //FileWriter writer;
        // predictions=new ArrayList();
        //PrintWriter printer =null; 
        
        try{              
            eval=new Evaluation(dadosTreino);

            eval.evaluateModel(modelo, dadosTeste); 
            
            //writer=new FileWriter(file,true);
            
            //printer=new PrintWriter(writer);
            
            System.out.println(eval.toSummaryString("===> Sumario estatistico <=== ",true));
            //printer.println(eval.toSummaryString("===> Sumario estatistico <=== ",true));
            //file.write(eval.toSummaryString("===> Sumario estatistico <=== ",true));
            System.out.println(eval.toClassDetailsString("===> Medidas de precisao <==="));
            //printer.println(eval.toClassDetailsString("===> Medidas de precisao <==="));
            //file.write(eval.toClassDetailsString("===> Medidas de precisao <===")+"\n\n");
            System.out.println(eval.toMatrixString("===> Matriz Confusão <==="));
            //printer.println(eval.toMatrixString("===> Matriz Confusão <==="));
            //file.write(eval.toMatrixString("===> Matriz Confusão <===")+"\n\n\n");
            //System.out.println("Recall: "+eval.recall(0)+"-"+eval.recall(1));
            //printer.println("Recall: "+eval.recall(0)+"-"+eval.recall(1));
            //predictions=eval.predictions();           

            return eval;
        } catch (Exception ex) {
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);
            return null;
        } /* finally{
            if (printer != null){
                printer.close();
            }
        }*/
    }
    
    
    /**
     * Function to split a specified dataset into training and test datasets
     * @param dataset - dataset to split
     * @param percentTraining - percent of dataset to be used as training data
     * @param seed - 
     * @return array with training data (index zero) and test data (index 1)
     */
    public static Instances[] geraTreinoTeste(Instances dataset, double percentTraining, int seed){
        Instances dados[],training,test;
        
        dados=new Instances[2];
        
        Resample filtro=new Resample();

        // 0.0 => distribuição da classe como está
        // 1.0 => classe com distribuição uniformizada
        filtro.setBiasToUniformClass(0.0);

        // Sem reposição das instâncias seleccionadas
        filtro.setNoReplacement(true);

        // Percentagem a amostrar
        filtro.setSampleSizePercent(percentTraining);

        // Seed
        filtro.setSeed(seed);
            
        try {    
            filtro.setInputFormat(dataset);       // prepare the filter for the data format
            
            filtro.setInvertSelection(false);  // do not invert the selection
            
            // apply filter
            training = Filter.useFilter(dataset, filtro);
            
            filtro.setInputFormat(dataset); // Re-initialize the filter, as per suggestion on Weka Mailing list
            
            filtro.setInvertSelection(true);  // invert the selection
            //percentTraining=100.0-percentTraining;
            //System.out.println("Percentagem de teste: "+percentTraining);
            //filtro.setSampleSizePercent(percentTraining);
            test = Filter.useFilter(dataset, filtro);

            System.out.println("Numero de linhas de teste: "+test.numInstances());
                     
            
            dados[0]=training;
            dados[1]=test;
            
            return dados;
        } catch (Exception ex) {
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);
            return null;
        }                        
    } 
    
    
    /**
     * Function to generate the array of strings containing configuration options to CLONALG algorithm
     * @param poolSize - Antibody pool size
     * @param clonalFactor - Clonal factor
     * @param numGen - number of generations
     * @param remainder - Remainder Pool Ratio
     * @param selectionPoolSize - Selection Pool Size
     * @param totalReplacement - Total Replacement
     * @param seed
     * @return Array de strings com opçoes
     */
    public static String[] geraOptCLONALG(int poolSize,double clonalFactor,int numGen,double remainder,
                                            int selectionPoolSize,int totalReplacement,int seed){
        String[] opt=new String[14];
        
        opt[0]="-N"; // Antibody Pool Size
        opt[1]=Integer.toString(poolSize);
        opt[2]="-B"; // Clonal Factor
        opt[3]=Double.toString(clonalFactor);
        opt[4]="-G"; // Number of Generations
        opt[5]=Integer.toString(numGen);
        opt[6]="-R"; // Remainder Pool Ratio
        opt[7]=Double.toString(remainder);
        opt[8]="-n"; // Selection Pool Size
        opt[9]=Integer.toString(selectionPoolSize);
        opt[10]="-D"; // Total Replacement
        opt[11]=Integer.toString(totalReplacement);
        opt[12]="-S"; // Seed
        opt[13]=Integer.toString(seed);
                                        
        return opt;
    }
    
    
    // Function to test a model using specified test data
    // Returns an ArrayList with the predictions
    /**
     * Function to test a model
     * @param modelo - modelo (classificador) a testar
     * @param dadosTreino - dados de Treino
     * @param dadosTeste - dados de Teste
     * @return - devolve um ArrayList com as predições (predictions)
     */
    public static ArrayList testaModelo(Classifier modelo,Instances dadosTreino,Instances dadosTeste){
        
        Evaluation eval;
        ArrayList predictions;

        predictions=new ArrayList();
            
        try {    
            eval=new Evaluation(dadosTreino);
            
            eval.evaluateModel(modelo, dadosTeste); 
            System.out.println(eval.toSummaryString("===> Sumario estatistico <=== ",true));
            System.out.println(eval.toClassDetailsString("===> Medidas de precisao <==="));
            System.out.println(eval.toMatrixString("===> Matriz Confusão <==="));
            predictions=eval.predictions();
            
            
        } catch (Exception ex) {
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);
            predictions=null;
        }
        
        return predictions;
        
    }
    
    /**
     * Function to display help message
     */
    static void showHelp(){
        System.out.println("comandos disponiveis:\n");
        System.out.println("training <algoritmo>  -> para treinar o algoritmos");
        System.out.println("test <algoritmo>  -> para testar os algoritmos");
        System.out.println("setup fich1 fich2  -> gera ficheiros de treino/teste e zeroday a partir dos ficheiros indicados");
        System.out.println("Os ficheiros correspondentes do dataset CIC-IDS2018 são identifcados apenas por 2 números de 2 algarismos (ex.: 02 03)");
        System.out.println("\n\n");
        System.out.println("Algoritmos disponiveis: \n");
        System.out.println("clonalg => Algoritmo CLONALG");
        System.out.println("mlp => Algoritmo Back-Propagation Multi-Layer Perceptron");
        System.out.println("lvq => Learning Vector Quantization");
        System.out.println("\n");
    }
    
    /**
     * Function that prepares dataset to be used (includes pre-processing tasks)
     * @param seed
     * @return 
     */
    public static Instances[] preparaDataset (int numAmostras,int seed){
        
        Instances dataset,dadosTreino,dadosTeste;
        
        Instances[] instances;
        
        System.out.println("=> Abre ficheiro");
        // Abre o ficheiro de treino
        dataset=abreDataset(ficheiro1);
        
        /* System.out.println("=> Trata Classe");
        // Trata a classe
        dataset=trataClasse(dataset);
        
        dataset=trataMissingValues(dataset); */
        
        dataset=preprocessaDataset(dataset);
        
        // Para efeitos de teste da ferramenta, vai buscar apenas 100000 amostras
        // No teste final, comentar as duas linhas seguintes
        System.out.println("=> Faz o Resample");
        dataset=divideDataset(dataset,numAmostras,seed);
        
        // gera dados de treino e dados de teste
        instances=geraTreinoTeste(dataset,70.0,seed);
        dadosTreino=instances[0];
        dadosTeste=instances[1];
        
        
        System.out.println("=====================================");
        System.out.println("Dados de Treino");

        //System.out.println(dados.toSummaryString());

        System.out.println("Valores distintos da classe: "+dadosTreino.numClasses());
        System.out.println("Numero de linhas do dataset: " + dadosTreino.attributeStats(dadosTreino.classIndex()).totalCount);

        // Attribute classe=dados.classAttribute();

        estatisticaClasse(dadosTreino);
        System.out.println("=====================================\n\n");
        
        
        System.out.println("=====================================");
        System.out.println("Dados de Teste");

        //System.out.println(dados.toSummaryString());

        System.out.println("Valores distintos da classe: "+dadosTeste.numClasses());
        System.out.println("Numero de linhas do dataset: " + dadosTeste.attributeStats(dadosTeste.classIndex()).totalCount);
        estatisticaClasse(dadosTeste);
        System.out.println("=====================================\n\n");
        
        return instances;
    }
    
    
    /**
     * Function to generate the 3 CLONALG models/classifiers used in my thesis
     * The generated models are saved in a default location for further use
     * @param dadosTreino
     * @param seed 
     */
    static void geraModelosClonalg(Instances dadosTreino, int seed){
        String [] optCLONALG1,optCLONALG2,optCLONALG3;
        Classifier modelo;        
        
                // Geração dos modelos CLONALG
        // Definição das diferentes options, que permitirão criar 3 modelos diferentes
        // As options são um String []
        
        /**
         * Modelo CLONALG número 1
         * 
         * Parametros
         * 
         * Antibody Pool Size: 120
         * Clonal Factor:0.3
         * Number of generations: 40  
         * Remainder Pool Ratio: 0.2
         * Selection Pool Size: 80
         * Total Replacement: 10
         * Seed: 1
         * 
         */
        optCLONALG1=geraOptCLONALG(120,0.3,40,0.2,80,10,seed);
        
        /**
         * Modelo CLONALG número 2
         * 
         * Parametros
         * 
         * Antibody Pool Size: 50
         * Clonal Factor:0.5
         * Number of generations: 40  
         * Remainder Pool Ratio: 0.2
         * Selection Pool Size: 20
         * Total Replacement: 10
         * Seed: 1
         * 
         */
        optCLONALG2=geraOptCLONALG(50,0.5,40,0.2,20,10,seed);
        
        /**
         * Modelo CLONALG número 3
         * 
         * Parametros
         * 
         * Antibody Pool Size: 40
         * Clonal Factor:0.3
         * Number of generations: 20  
         * Remainder Pool Ratio: 0.4
         * Selection Pool Size: 20
         * Total Replacement: 15
         * Seed: 1
         * 
         */
        optCLONALG3=geraOptCLONALG(40,0.3,20,0.4,20,15,seed);
        
        // Conjunto das opções arrumado num array; facilita a automatização e a geração dos 3 modelos
        String[][] optCLONALG={optCLONALG1,optCLONALG2,optCLONALG3};
        
        // Gera os 3 modelos, com um ciclo
        try {
            for (int i=0; i<optCLONALG.length;i++){

                System.out.println("Gerando o modelo CLONALG"+Integer.toString(i));

                modelo=geraModeloCLONALG(dadosTreino,optCLONALG[i]);

                String nomefich1=modelos_path+"modCLONALG"+Integer.toString(i);

                // Guarda o modelo
                SerializationHelper.write(nomefich1,modelo);

            }
        } catch (Exception ex){
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);            
        }                
        
    }
    
    
    
    
    /**
     * Function to test the 3 CLONALG generated models using the specified test data
     * @param dadosTreino
     * @param dadosTeste
     * @param seed 
     */
    public static void testaModelosClonalg(Instances dadosTreino,Instances dadosTeste, int seed){
        
        Vote ensemble;                
        
        ArrayList predictions;                
        
        try {
        // Le os modelos e avalia-os com os dados de teste
            for (int i=0;i<numModelos;i++){
                // Lê o modelo
                String nomefich=modelos_path+"modCLONALG"+Integer.toString(i);
                Classifier cls=(Classifier)weka.core.SerializationHelper.read(nomefich);

                // predictions... para tratamento futuro
                predictions=testaModelo(cls,dadosTreino,dadosTeste);
                
                geraROCCurve(predictions);
            }  
        } catch (Exception ex){
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);            
        }
        
        
        ensemble=geraEnsembleCLONALG(seed);                
        
        System.out.println("Testando o Classificador Ensemble");
        predictions=testaModelo(ensemble,dadosTreino,dadosTeste);
        
        geraROCCurve(predictions);
        
        
    }
    
    /**
     * Function to generate an ensemble from the 3 pre-generated CLONALG models
     * The Majority Voting Strategy was used
     * @param seed
     * @return o modelo ensemble
     */
    public static Vote geraEnsembleCLONALG(int seed){
        
        Vote ensemble;
        // Majority Voting
        ensemble = new Vote();
        
        SelectedTag tag = new SelectedTag(Vote.MAJORITY_VOTING_RULE,Vote.TAGS_RULES);
        ensemble.setCombinationRule(tag);
        
        ensemble.setSeed(seed);
        
        File[] preBuiltClassifiers=new File[3];
        
        System.out.println("Gerando o Classificador Ensemble");
        
        try {
            for (int i=0;i<numModelos;i++){
                String nome=modelos_path+"modCLONALG"+Integer.toString(i);
                Classifier cls=(Classifier)weka.core.SerializationHelper.read(nome);
                //preBuiltClassifiers[i]=new File(nome);

                System.out.println("Adicionando o modelo "+nome);

                ensemble.addPreBuiltClassifier(cls);
            }
            return (ensemble);
                                    
        } catch (Exception ex){
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);
            return (null);
        }
    }
    
    
    /**
     * Function to test the 3 pre-generated CLONALG models/classifiers against a data stored in a file
     * The purpose is to simulate zero-day attack detection
     * @param seed 
     */
    public static void zeroDayCLONALG(int seed){
        
        Vote ensemble;
        // Abre segundo ficheiro de dados e usa-o  como dataset de testes
         
        Instances dataset;
        ArrayList predictions;
        
        System.out.println("=> Abre ficheiro de dados");
        // Abre o ficheiro de treino
        dataset=abreDataset(zeroday_file);
        
        estatisticaClasse(dataset);
        
        /*
        System.out.println("=> Trata Classe");
        // Trata a classe
        dataset=trataClasse(dataset);
        
        estatisticaClasse(dataset);
        
        dataset=trataMissingValues(dataset); 
        
        // Para efeitos de teste da ferramenta, vai buscar apenas 100000 amostras
        // No teste final, comentar as duas linhas seguintes
        // System.out.println("=> Faz o Resample");
        // dataset=divideDataset(dataset,100000,seed); */
        
        try{
        // Le os modelos e avalia-os com os dados de teste
            for (int i=0;i<numModelos;i++){
                // Lê o modelo
                String nomefich=modelos_path+"modCLONALG"+Integer.toString(i);
                Classifier cls=(Classifier)weka.core.SerializationHelper.read(nomefich);

                // predictions... para tratamento futuro
                predictions=testaModelo(cls,dataset,dataset);
                
                geraROCCurve(predictions);
            }
        } catch (Exception ex) {
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);
        }
                
        ensemble=geraEnsembleCLONALG(seed);
        
        System.out.println("Testando o Classificador Ensemble");
        predictions=testaModelo(ensemble,dataset,dataset);
        
        geraROCCurve(predictions);
        
        //System.exit(0);
    }
    
    /**
     * Function to generate ROC curves based on the specified predictions
     * The code is based on https://waikato.github.io/weka-wiki/generating_roc_curve/
     * @param predictions 
     */
    public static void geraROCCurve(ArrayList predictions){
        // generate curve
        ThresholdCurve tc = new ThresholdCurve();
        int classIndex = 0;
        Instances result = tc.getCurve(predictions, classIndex);

        // plot curve
        ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
        vmc.setROCString("(Area under ROC = " + Utils.doubleToString(tc.getROCArea(result), 4) + ")");
        vmc.setName(result.relationName());
        PlotData2D tempd = new PlotData2D(result);
        tempd.setPlotName(result.relationName());
        tempd.addInstanceNumberAttribute();
        
        // specify which points are connected
        boolean[] cp = new boolean[result.numInstances()];
        for (int n = 1; n < cp.length; n++)
            cp[n] = true;
        
        try{
            tempd.setConnectPoints(cp);

            // add plot
            vmc.addPlot(tempd);

        } catch (Exception ex){
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);
        }
        
        // display curve
        String plotName = vmc.getName();
        final javax.swing.JFrame jf = new javax.swing.JFrame("Weka Classifier Visualize: "+plotName);
        jf.setSize(500,400);
        jf.getContentPane().setLayout(new BorderLayout());
        jf.getContentPane().add(vmc, BorderLayout.CENTER);
        jf.addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent e) {
                jf.dispose();
            }
        });
        jf.setVisible(true);
    }
    
    /**
     * Function to prepare test environment
     * Selects 2 datafiles from the original dataset (specified in command line - see below)
     * From the first one, extracts teh specified number of samples, splits it into training and test data
     * and saves the result in the training and test datafiles;
     * The second file is pre-processed and saved as zeroday.csv , to be further used as source to simulate a zero-day attack
     * 
     * file1 and file2 should be numbers (ex.: 02 03) and are command line arguments
     * 
     * @param file1 Ficheiro que será utilizado no treino do modelo
     * @param file2 Ficheiro utilizado para simular ataque zero-day
     * @param samples Numero de amostras do ficheiro file1 a considerar para a fase de treino
     * @param seed
     */
    public static void setup(String file1,String file2,int samples,int seed){
        
        Instances dadosTreino,dadosTeste,zeroDay,dataset;
        
        Instances[] instances;
        
        // Trata zeroday
        System.out.println("* * * Gerando ficheiro Zero-Day * * *");
        file2=dataset_path+"Dados_"+file2+".csv";
        
        System.out.println("A abrir o ficheiro "+file2);
        zeroDay=abreDataset(file2);
        
        System.out.println("=> Pre-processamento...");
        zeroDay=preprocessaDataset(zeroDay);
        
        System.out.println("Guarda o ficheiro Zero-Day");
        guardaDataset(zeroDay,zeroday_file);
        
        
        
        // Trata treino e teste
        System.out.println("* * * Gerando ficheiros de treino e teste * * *");
        
        file1=dataset_path+"Dados_"+file1+".csv";
        
        System.out.println("A abrir o ficheiro "+file1);
        dataset=abreDataset(file1);
        
        System.out.println("=> Pre-processamento...");
     
        dataset=preprocessaDataset(dataset);
        
        System.out.println("=> Faz o Resample");
        dataset=divideDataset(dataset,samples,seed);
        
        // gera dados de treino e dados de teste
        instances=geraTreinoTeste(dataset,70.0,seed);
        dadosTreino=instances[0];
        dadosTeste=instances[1];
        
        System.out.println("Guarda os ficheiros de treino e teste");
        guardaDataset(dadosTreino,ficheiro1_training);
        guardaDataset(dadosTeste,ficheiro1_test);   
    }
    
    /**
     * Function that does pre-processing on a specified dataset - class reduction and dealing with missing values
     * @param dataset
     * @return 
     */
    public static Instances preprocessaDataset(Instances dataset){
        System.out.println("Trata classe");        
        dataset=trataClasse(dataset);
        
        System.out.println("Trata Missing Values");
        dataset=trataMissingValues(dataset);
        
        System.out.println("Normalizando os dados");
        dataset=normalizaDados(dataset);
        
        return dataset;
    }
    
    /**
     * Function to save a specified dataset to disk (.CSV format)
     * @param dataset
     * @param filename 
     */
    public static void guardaDataset(Instances dataset,String filename){
        CSVSaver saver=new CSVSaver();
        
        saver.setInstances(dataset);
        try{
            saver.setFile(new File(filename));
            saver.writeBatch();
        } catch (Exception ex){
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);
        }        
    }
    
    /**
     * Função para criar um ensemble com dois algoritmos e testar o modelo resultante
     * Vai buscar os ficheiros de dados à nova estrutura...
     * @param seed 
     */
    public static void testaEnsembleDual(String alg1,String alg2,int seed){
        Vote ensemble;
        
        Instances dadosTeste;
        
        ArrayList predictions;
        
        String nome;
        
        Classifier cls;
        
        ensemble = new Vote();
        
        SelectedTag tag = new SelectedTag(Vote.MAJORITY_VOTING_RULE,Vote.TAGS_RULES);
        ensemble.setCombinationRule(tag);
        
        ensemble.setSeed(seed);
        
        //File[] preBuiltClassifiers=new File[3];
        
        System.out.println("Gerando o Classificador Ensemble "+alg1+"-"+alg2);
        
        try {
            for (int i=0;i<numModelos;i++){
                nome=modelos_path+"mod"+alg1+Integer.toString(i);
                cls=(Classifier)weka.core.SerializationHelper.read(nome);
                //preBuiltClassifiers[i]=new File(nome);

                System.out.println("Adicionando o modelo "+nome);

                ensemble.addPreBuiltClassifier(cls);                              
            }
            
            for (int i=0;i<numModelos;i++){                                
                nome=modelos_path+"mod"+alg2+Integer.toString(i);
                cls=(Classifier)weka.core.SerializationHelper.read(nome);
                //preBuiltClassifiers[i]=new File(nome);

                System.out.println("Adicionando o modelo "+nome);

                ensemble.addPreBuiltClassifier(cls);
            }
            // return (ensemble);
                                    
        } catch (Exception ex){
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);
            //return (null);
        }
        
        System.out.println("=> Testando o ensemble com dados de Teste");
        
        dadosTeste=abreDataset(ficheiro1_test);
        //dadosTeste=normalizaDados(dadosTeste);
        
        predictions=testaModelo(ensemble,dadosTeste,dadosTeste);
        
        System.out.println("=> Testando o ensemble com Zero-Day");
        
        dadosTeste=abreDataset(zeroday_file);
        //dadosTeste=normalizaDados(dadosTeste);
        
        predictions=testaModelo(ensemble,dadosTeste,dadosTeste);
                             
    }
    
    /**
     * Função que realiza todos os testes com ensembles de multiplos algoritmos
     * @param seed 
     */
    public static void testaEnsemble(int seed){
        
        // Testa ensemble CLONALG e MLP
        testaEnsembleDual("CLONALG","BackMLP",seed);
        
        // Testa Ensemble CLONALG e LVQ
        testaEnsembleDual("CLONALG","LVQ",seed);
        
        // Testa ensemble MLP e LVQ
        testaEnsembleDual("LVQ","BackMLP",seed);
        
        // Testa ensemble CLONALG, MLP E LVQ
        //testaEnsembleCLONALGMPLVQ(seed);
        
    }
    
    /**
     * Função que testa os modelos contra um ataque zero-day
     * @param alg algoritmo a testar; um de CLONALG,BackMLP e LVQ
     * @param seed 
     */
    public static void zeroDayAttack(String alg,int seed){
        Vote ensemble;
        // Abre segundo ficheiro de dados e usa-o  como dataset de testes
         
        Instances dataset;
        ArrayList predictions;
        
        System.out.println("=> Abre ficheiro de dados");
        // Abre o ficheiro de treino
        dataset=abreDataset(zeroday_file);
        
        estatisticaClasse(dataset);
        
        /* System.out.println("=> Trata Classe");
        // Trata a classe
        dataset=trataClasse(dataset);
        
        estatisticaClasse(dataset);
        
        dataset=trataMissingValues(dataset); */
        
        //dataset=normalizaDados(dataset);
        
        // Para efeitos de teste da ferramenta, vai buscar apenas 100000 amostras
        // No teste final, comentar as duas linhas seguintes
        // System.out.println("=> Faz o Resample");
        // dataset=divideDataset(dataset,100000,seed);
        
        try{
        // Le os modelos e avalia-os com os dados de teste
            for (int i=0;i<numModelos;i++){
                // Lê o modelo
                String nomefich=modelos_path+"mod"+alg+Integer.toString(i);
                Classifier cls=(Classifier)weka.core.SerializationHelper.read(nomefich);

                // predictions... para tratamento futuro
                predictions=testaModelo(cls,dataset,dataset);
                
                geraROCCurve(predictions);
            }
        } catch (Exception ex) {
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);
        }
                
        ensemble=geraEnsemble(alg,seed);
        
        System.out.println("Testando o Classificador Ensemble");
        predictions=testaModelo(ensemble,dataset,dataset);
        
        geraROCCurve(predictions);

    }
    
    /**
     * Função para gerar o ensemble de 3 modelos de um algoritmo
     * Os algoritmos são: CLONALG,BackMLP e LVQ
     * @param alg
     * @param seed
     * @return 
     */
    public static Vote geraEnsemble(String alg,int seed){
        Vote ensemble;
        // Majority Voting
        ensemble = new Vote();
        
        SelectedTag tag = new SelectedTag(Vote.MAJORITY_VOTING_RULE,Vote.TAGS_RULES);
        ensemble.setCombinationRule(tag);
        
        ensemble.setSeed(seed);
        
        File[] preBuiltClassifiers=new File[3];
        
        System.out.println("Gerando o Classificador Ensemble "+ alg);
        
        try {
            for (int i=0;i<numModelos;i++){
                String nome=modelos_path+"mod"+alg+Integer.toString(i);
                Classifier cls=(Classifier)weka.core.SerializationHelper.read(nome);
                //preBuiltClassifiers[i]=new File(nome);

                System.out.println("Adicionando o modelo "+nome);

                ensemble.addPreBuiltClassifier(cls);
            }
            return (ensemble);
                                    
        } catch (Exception ex){
            Logger.getLogger(BenchmarkIDS.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Error: " + ex);
            System.exit(1);
            return (null);
        }
    }
    
    /**
     * Função para executar o cenário 1
     * Não é necessário pré-processar os dados, pois já veem processados do fluxo de geração
     * @param cenario - numero do cenario
     * @param dados1 - ficheiro de dados para treino
     * @param dados2 - ficheiro de dados para teste
     * @throws java.io.IOException
     */
    public static void cenarios(int cenario,String dados1,String dados2) throws IOException,Exception{
        
        Instances dadosTreino,dadosTeste,treino,teste;
        Classifier clonalg,mlp,lvq,cls;
        Vote ensemble;
        Evaluation evalCLONALG,evalMLP,evalLVQ,evalEnsemble;
        
        //Instant start,finish;
        //long tempo;
        double accuracy;
        
        String [] optCLONALG,optMLP,optLVQ;
        
        //String dados1="C:\\Developer\\Dados4Testes\\Dia1NormAt1.csv";
        //String dados2="C:\\Developer\\Dados4Testes\\Dia1NormAt2.csv";
        String report="C:\\Developer\\Dados4Testes\\Final\\Reports16_21\\BenchIDSCenario"+Integer.toString(cenario)+".txt";
        String modelosPath="C:\\Developer\\Dados4Testes\\Modelos\\";
        String linha;
        
        // Indice da classe para tráfego malicioso (o indice é zero-based)
        int classeMalicioso=1;
        
        //File file=new File(report);
        FileWriter fwriter=new FileWriter(report);
        //PrintWriter pw=null;
        
        // Para gerar dados de treino e teste diferentes, para garantir variedade e
        // significancia estatística
        int[] seeds={1,5,10,23,34,47,55,69,88,93};
        
        System.out.println("====> Início: "+LocalDateTime.now()); 
        
        System.out.println("Abrindo os ficheiros de dados \n");
        dadosTreino=abreDataset(dados1);
        dadosTeste=abreDataset(dados2);
        
        // Escreve Cabeçalho do ficheiro
        fwriter.write("Algoritmo;Seed;TrueNegative;FalsePositive;FalseNegative;TruePositive;TruePositiveRate;FalsePositiveRate;TrueNegativeRate;FalseNegativeRate;Precision;Recall;Accuracy;F1\n");
        for (int seed: seeds){
            
            System.out.println("====> Valor do Seed: "+seed);
            //fwriter.write("====> Valor do Seed: "+seed+"\n\n");
            
            // Total de 200k linhas
            // 70% para treino (140000)
            // 30% para teste (60000)
            treino=divideDataset(dadosTreino,140000,seed);
            teste=divideDataset(dadosTeste,60000,seed);
            
            // As opções dos modelos foram seleccionadas com base nos testes
            // feitos atraves deata aplicação (opções training <algoritmo> e test <algoritmo>
            
            // Opções CLONALG
            optCLONALG=geraOptCLONALG(120,0.3,40,0.2,80,10,seed);
            
            // Opções MLP
            optMLP=geraOptBackMLP("a",0.1,0.3,500,seed);
            
            // Opções LVQ
            //optLVQ=geraOptLVQ3(0.1,1,1,0.3,50,2500,true,0.2,seed); opções erradas...
            optLVQ=geraOptLVQ3(0.1,1,1,0.3,90,4500,false,0.2,seed);
            
            System.out.println("===> Modelo CLONALG<===");
            // Geração e teste do modelo CLONALG
            clonalg=geraModeloCLONALG(treino,optCLONALG);                        
            
            evalCLONALG=testaModeloEvaluation(clonalg,treino,teste); 
            
            // Escreve dados no ficheiro
            accuracy=calculaAccuracy(evalCLONALG,classeMalicioso);
            linha=constroiLinha("CLONALG",seed,evalCLONALG,accuracy,classeMalicioso);            
            fwriter.write(linha);
            
            System.out.println("===> Modelo LVQ<===");
            //fwriter.write("====> Algoritmo: LVQ  => Seed: "+seed+"\n\n");
            // Geração e teste do modelo LVQ
            
            //start=Instant.now();
            lvq=geraModeloLVQ(treino,optLVQ);
            evalLVQ=testaModeloEvaluation(lvq,treino,teste);
            //finish=Instant.now();
            //tempo=Duration.between(start, finish).toMinutes();
            //System.out.println("Tempo de execução: "+ tempo+" minutos");                        
            
            // Escreve dados no ficheiro
            accuracy=calculaAccuracy(evalLVQ,classeMalicioso);
            linha=constroiLinha("LVQ",seed,evalLVQ,accuracy,classeMalicioso);            
            fwriter.write(linha);
            
            System.out.println("===> Modelo MLP<===");
            // Geração e teste do modelo MLP
            mlp=geraModeloBackMLP(treino,optMLP);
            evalMLP=testaModeloEvaluation(mlp,treino,teste);                         
            
            // Escreve dados no ficheiro
            accuracy=calculaAccuracy(evalMLP,classeMalicioso);
            linha=constroiLinha("MLP",seed,evalMLP,accuracy,classeMalicioso);            
            fwriter.write(linha);
            
            // Ensemble
            
            System.out.println("===> Ensemble LVQ+CLONALG+MLP <===");
            ensemble = new Vote();
        
            SelectedTag tag = new SelectedTag(Vote.MAJORITY_VOTING_RULE,Vote.TAGS_RULES);
            ensemble.setCombinationRule(tag);
        
            ensemble.setSeed(seed);                    
        
            System.out.println("Gerando o Classificador Ensemble");
            
            // Adiciona CLONALG          
            ensemble.addPreBuiltClassifier(clonalg);
            
            // Adiciona LVQ           
            ensemble.addPreBuiltClassifier(lvq);
            
            // Adiciona MLP            
            ensemble.addPreBuiltClassifier(mlp);
            
            evalEnsemble=testaModeloEvaluation(ensemble,treino,teste); 
            
            // Escreve dados no ficheiro
            accuracy=calculaAccuracy(evalEnsemble,classeMalicioso);
            linha=constroiLinha("Ensemble",seed,evalEnsemble,accuracy,classeMalicioso);            
            fwriter.write(linha);
                        
        }
        fwriter.close();
        System.out.println("====> Fim: "+LocalDateTime.now());
    }
    

    /**
     * Função para calcular a métrica Accuracy
     * @param eval
     * @param classIndex
     * 
     * @return 
     */
    public static double calculaAccuracy(Evaluation eval,int classIndex){
        double acc;
        
        double tp,tn,fp,fn;
        
        tp=eval.numTruePositives(classIndex);
        tn=eval.numTrueNegatives(classIndex);
        fp=eval.numFalsePositives(classIndex);
        fn=eval.numFalseNegatives(classIndex);
        
        acc=(tp + tn)/(tp+tn+fp+fn);
        
        return acc;
    }
    
    
    /**
     * Constroi Linha a inserir no ficheiro txt
     * @param algoritmo
     * @param seed
     * @param eval
     * @param acc
     * @param classIndex
     * @return 
     */
    public static String constroiLinha(String algoritmo,int seed,Evaluation eval,double acc,int classIndex){
        
        String linha;
        String sep=";";  // separador
        linha=algoritmo+sep+seed+sep+eval.numTrueNegatives(classIndex)+sep+eval.numFalsePositives(classIndex)+sep;
        linha+=eval.numFalseNegatives(classIndex)+sep+eval.numTruePositives(classIndex)+sep;
        linha+=eval.truePositiveRate(classIndex)+sep+eval.falsePositiveRate(classIndex)+sep;
        linha+=eval.trueNegativeRate(classIndex)+sep+eval.falseNegativeRate(classIndex)+sep;
        linha+=eval.precision(classIndex)+sep+eval.recall(classIndex)+sep+acc+sep+eval.fMeasure(classIndex)+"\n";
        
        // substitui ponto decimal por vírgula
        linha=linha.replace('.', ',');
                
        return linha;
    }
    
    /**
     * 
     * 
     * MAIN FUNCTION
     * 
     * 
     */
    
    
    /**
     * @param args the command line arguments
     * @throws Exception
     */
    public static void main(String[] args) throws Exception{
        // TODO code application logic here
        Scanner input;
        
        Instances dataset,dadosTreino,dadosTeste;
        Instances[] instances;
        
        ArrayList predictions;
                
        Classifier modelo;
        
        String dados1,dados2;
        
        // Variáveis para opções dos modelos CLONALG
        //String[] optCLONALG1,optCLONALG2,optCLONALG3;
        
        
        
        
        
        /**
         * Global seed value
         */
        int globalSeed=1;
        
        
        // Trata argumentos da linha de comandos
        if (args.length<1){
            showHelp();
            System.exit(0);
        }
        
        switch (args[0]){
            case "training":
                switch (args[1]){
                    case "clonalg":
                        // Gera CLONALG
                        System.out.println("Gerando os modelos CLONALG");
                        //instances=preparaDataset(200000,globalSeed);
                        dadosTreino=abreDataset(ficheiro1_training);
                
                        //dadosTreino=instances[0];
                        // dadosTeste=instances[1]; // Not needed
                
                        geraModelosClonalg(dadosTreino,globalSeed);
                                
                        System.exit(0);
                        
                        break;
                    case "mlp":
                        // Gera BackMLP
                        System.out.println("Gerando os modelos BackMLP");
                        // instances=preparaDataset(200000,globalSeed);
                        dadosTreino=abreDataset(ficheiro1_training);
                        //dadosTreino=instances[0];
                        // dadosTeste=instances[1]; // Not needed
                
                        geraModelosBackMLP(dadosTreino,globalSeed);
                                
                        System.exit(0);
                        break;
                    case "lvq":
                        // Gera LVQ
                        System.out.println("Gerando os modelos LVQ3");
                        // instances=preparaDataset(200000,globalSeed);
                        dadosTreino=abreDataset(ficheiro1_training);
                
                        //dadosTreino=instances[0];
                        // dadosTeste=instances[1]; // Not needed
                
                        geraModelosLVQ3(dadosTreino,globalSeed);
                                
                        System.exit(0);
                        break;
                }
                break;
            case "test":
                switch (args[1]){
                    case "clonalg":
                        // Testa CLONALG
                        System.out.println("Testando os modelos CLONALG");
                                 
                        //instances=preparaDataset(200000,globalSeed); 
                        //dadosTreino=abreDataset(ficheiro1_training);
                        //dadosTreino=instances[0];
                        //dadosTeste=instances[1];
                        dadosTeste=abreDataset(ficheiro1_test);
                        //dadosTeste=normalizaDados(dadosTeste);
                
                        testaModelosClonalg(dadosTeste,dadosTeste,globalSeed);
                
                        zeroDayAttack("CLONALG",globalSeed);
                
                        input = new Scanner(System.in);
                        System.out.print("Press Enter to quit...");
                        input.nextLine();
                
                        System.exit(0);
                        break;
                    case "mlp":
                        // Testa BackMLP
                        System.out.println("Testando os modelos BackMLP");
                                 
                        /* instances=preparaDataset(200000,globalSeed); 
                        dadosTreino=instances[0];
                        dadosTeste=instances[1];
                        
                        dadosTreino=normalizaDados(dadosTreino);*/ 
                        
                        dadosTeste=abreDataset(ficheiro1_test);
                        
                        //dadosTeste=normalizaDados(dadosTeste);
                
                        testaModelosBackMLP(dadosTeste,dadosTeste,globalSeed);
                        
                        System.out.println("Testando os modelos BackMLP com Zero-Day");
                        
                        zeroDayAttack("BackMLP",globalSeed);
                
                        input = new Scanner(System.in);
                        System.out.print("Press Enter to quit...");
                        input.nextLine();
                
                        System.exit(0);
                        break;
                    case "lvq":
                        // Testa LVQ
                        System.out.println("Testando os modelos LVQ");
                                 
                        /* instances=preparaDataset(200000,globalSeed); 
                        dadosTreino=instances[0];
                        dadosTeste=instances[1];
                        
                        dadosTreino=normalizaDados(dadosTreino); */
                        dadosTeste=abreDataset(ficheiro1_test);
                        //dadosTeste=normalizaDados(dadosTeste);
                
                        testaModelosLVQ(dadosTeste,dadosTeste,globalSeed);
                        
                        System.out.println("Testando os modelos LVQ com Zero-Day");
                        
                        zeroDayAttack("LVQ",globalSeed);
                
                        input = new Scanner(System.in);
                        System.out.print("Press Enter to quit...");
                        input.nextLine();
                
                        System.exit(0);
                        break;
                }
                break;
            case "ensemble":
                testaEnsemble(globalSeed);
                break;
            case "setup":
                if (args.length<3){
                    showHelp();
                }
                String fich1=args[1];
                String fich2=args[2];
                setup(fich1,fich2,200000,globalSeed);
                break;
            case "cenario1":
                dados1="C:\\Developer\\Dados4Testes\\Dia16NormAt1.csv";
                dados2="C:\\Developer\\Dados4Testes\\Dia16NormAt2.csv";
                cenarios(1,dados1,dados2);
                break;
            case "cenario2":
                dados1="C:\\Developer\\Dados4Testes\\Dia16NormAt1.csv";
                dados2="C:\\Developer\\Dados4Testes\\Dia2NormAt1.csv";
                cenarios(2,dados1,dados2);
                break;
            case "cenario3":
                dados1="C:\\Developer\\Dados4Testes\\Dia16NormAt2.csv";
                dados2="C:\\Developer\\Dados4Testes\\Dia2NormAt2.csv";
                cenarios(3,dados1,dados2);
                break;
            case "cenario4":
                dados1="C:\\Developer\\Dados4Testes\\Dia16NormAtaques.csv";
                dados2="C:\\Developer\\Dados4Testes\\Dia2NormAtaques.csv";
                cenarios(4,dados1,dados2);
                break;
            case "cenario5":
                dados2="C:\\Developer\\Dados4Testes\\Dia16NormAtaques.csv";
                dados1="C:\\Developer\\Dados4Testes\\Dia2NormAtaques.csv";
                cenarios(5,dados1,dados2);
                break;
            case "cenario6":
                dados2="C:\\Developer\\Dados4Testes\\Dia15NormAtaques.csv";
                dados1="C:\\Developer\\Dados4Testes\\Dia2NormAtaques.csv";
                cenarios(6,dados1,dados2);
                break;
            default:
                showHelp();
                System.exit(0);
        }
        
/*        if (args[0].equalsIgnoreCase("training")){
            if (args[1].equalsIgnoreCase("clonalg")){
                // Gera CLONALG
                System.out.println("Gerando os modelos CLONALG");
                instances=preparaDataset(200000,globalSeed);
                
                dadosTreino=instances[0];
                // dadosTeste=instances[1]; // Not needed
                
                geraModelosClonalg(dadosTreino,globalSeed);
                                
                System.exit(0);
            } else if (args[1].equalsIgnoreCase("mlp")){
                // Gera CLONALG
                System.out.println("Gerando os modelos BackMLP");
                instances=preparaDataset(200000,globalSeed);
                
                dadosTreino=instances[0];
                // dadosTeste=instances[1]; // Not needed
                
                geraModelosBackMLP(dadosTreino,globalSeed);
                                
                System.exit(0);
            } else {
                showHelp();
                System.exit(0);
            }
        } else if (args[0].equalsIgnoreCase("test")){
            if (args[1].equalsIgnoreCase("clonalg")){
                // Testa CLONALG
                System.out.println("Testando os modelos CLONALG");
                
                 
                instances=preparaDataset(200000,globalSeed); 
                dadosTreino=instances[0];
                dadosTeste=instances[1];
                
                testaModelosClonalg(dadosTreino,dadosTeste,globalSeed);
                
                zeroDayCLONALG(globalSeed);
                
                Scanner input = new Scanner(System.in);
                System.out.print("Press Enter to quit...");
                input.nextLine();
                
                System.exit(0);
            } else {
                showHelp();
                System.exit(0);
            }
        } else {
            showHelp();
            System.exit(0);
        } */
        
        
    }
            
}
