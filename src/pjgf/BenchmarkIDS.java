/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package pjgf;

import java.awt.BorderLayout;
import java.io.File;
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
import weka.classifiers.immune.clonalg.CLONALG;
import weka.classifiers.meta.Vote;
import static weka.classifiers.neural.common.learning.LearningKernelFactory.LEARNING_FUNCTION_STATIC;
import static weka.classifiers.neural.common.learning.LearningKernelFactory.TAGS_LEARNING_FUNCTION;
import static weka.classifiers.neural.common.training.TrainerFactory.TAGS_TRAINING_MODE;
import static weka.classifiers.neural.common.training.TrainerFactory.TRAINER_BATCH;
import static weka.classifiers.neural.common.transfer.TransferFunctionFactory.TAGS_TRANSFER_FUNCTION;
import static weka.classifiers.neural.common.transfer.TransferFunctionFactory.TRANSFER_SIGMOID;
import weka.classifiers.neural.multilayerperceptron.BackPropagation;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
// import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

/**
 *
 * @author Paulo User
 */
public class BenchmarkIDS {
    
    // Constantes
    private static final String data_path="c:\\Developer\\dados_mcif\\";
    private static final String modelos_path="c:\\Developer\\dados_mcif\\modelos\\";
    private static final int numModelos=3;
    
    // Ficheiro com os dados de treino e teste para gerar modelos
    private static final String ficheiro1=data_path+"Dados_01.csv";
        
    // Ficheiro com os dados de teste
    private static final String ficheiro2=data_path+"Dados_05.csv";
    
    /**
     * Função para abrir o dataset a partir de um ficheiro
     * @param filename
     * @return 
     * 
     */
    public static Instances abreDataset(String filename) {
       
        Instances dataset;
            
        try {    
            DataSource ficheiro=new DataSource(filename);
            dataset=ficheiro.getDataSet();
            
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
    
    
    // Obtem numero de linhas para cada valor distinto da classe
    // O dataset tem que ter a classe definida
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
     * Função que, a partir de um dataset, devolve uma amostra com,
     * aproximadamente, numLinhas linhas; os dados estão estratificados (stratified)
     * @param dados - O dataset a dividir
     * @param numLinhas - número de linhas pretendidas
     * @param seed - seed para utilizar
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
    
    
    // Função para alterar as classes, para contemplar apenas 2 valores:
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
     * Função para tratar os Missing Values (NaN) do dataset
     * 
     * @param dados - dataset a tratar
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
     * Gera o modelo com baso no algoritmo CLONALG
     * @param dadosTreino : dataset para treino do modelo
     * @param options : parametros do modelo
     * @return Retorna um modelo clonalg, cuja configuração é dada pelas options
     * e treinado com os dadosTreino
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
    
    public static String[] geraOptBackMLP(int layer1,int layer2,int layer3,
            double bias,double learningRate,int learningRateFunction,double momentum,
            int iterations,int transfer,double weightDecay,int seed){
        
        String[] opt=new String[22];
        
        opt[0]="-X"; // hidden layer 1
        opt[1]=Integer.toString(layer1);
        opt[2]="-Y"; // hidden layer 2
        opt[3]=Double.toString(layer2);
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
     * GeraModeloBackMLP - Gera um modelo recorrendo a back-propagation MLP
     * 
     * @param dadosTreino - Dados de Treino
     * @param options - opcções de configuração do modelo
     * @return 
     *  
     */
     public static Classifier geraModeloBackMLP(Instances dadosTreino,String [] options) {
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
     * Função para gerar os 3 modelos BackMLP
     * A função grava os 3 modelos no directório pré-definido
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
        backMLP1=geraOptBackMLP(80,40,10,1.0,0.1,3,0.0,500,1,0.0,seed);
        
        
        /* Opções do modelo BackMLP2 - altera transfer function e momentum
            layer1: 80
            layer2: 40
            layer3: 10
            biasInput: 1.0
            learningRate: 0.1
            learningRateFunction: 3
            momentum: 0.3
            iterations: 500
            transferFunction: 2
            weightDecay: 0.0
        */                
        backMLP2=geraOptBackMLP(80,40,10,1.0,0.1,3,0.3,500,2,0.0,seed);
        
        
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
        backMLP3=geraOptBackMLP(100,80,40,1.0,0.1,3,0.3,500,1,0.2,seed);
        
        
        // Conjunto das opções arrumado num array; facilita a automatização e a geração dos 3 modelos
        String[][] optBackMLP={backMLP1,backMLP2,backMLP3};
        
        // Normaliza os dados; necessário para BackMLP...
        
        // Gera os 3 modelos, com um ciclo
        try {
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
     * Função para testar os 3 modelos BackMLP
     * @param dadosTreino
     * @param dadosTeste
     * @param seed 
     */
    public static void testaModelosBackMLP(Instances dadosTreino,Instances dadosTeste,int seed){
        Vote ensemble;                
        
        ArrayList predictions;                
        
        try {
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
     * Função para gerar o ensemble dos 3 modelos BackMLP
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
    
    public static void zeroDayBackMLP(int seed){
        Vote ensemble;
        // Abre segundo ficheiro de dados e usa-o  como dataset de testes
         
        Instances dataset;
        ArrayList predictions;
        
        System.out.println("=> Abre ficheiro de dados");
        // Abre o ficheiro de treino
        dataset=abreDataset(ficheiro2);
        
        estatisticaClasse(dataset);
        
        System.out.println("=> Trata Classe");
        // Trata a classe
        dataset=trataClasse(dataset);
        
        estatisticaClasse(dataset);
        
        dataset=trataMissingValues(dataset); 
        
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
     
     
     // Testa o modelo (dados de teste) e devolve predicoes
    // A ideia é devolver um Evaluation, que possa, depois, ser tratada no
    // programa principal
    /**
     * Função para testar um modelo (classificador)
     * @param modelo - modelo a testar
     * @param dadosTreino - dados de treino
     * @param dadosTeste - dados de teste
     * @return  - devolve a Evaluation
     */
    public static Evaluation testaModeloEvaluation(Classifier modelo,Instances dadosTreino,Instances dadosTeste) {
        
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
    
    
    /**
     * Função para dividir um dataset em dados de treino e dados de teste
     * @param dataset - dataset a dividir
     * @param percentTraining - percentagem do dataset a ser considerado dados de treino
     * @param seed - 
     * @return array com dados de treino (posição zero) e dados de teste (posição 1)
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
     * Gera o array de strings com as opções para o CLONALG
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
    
    
    // Testa o modelo (dados de teste) e devolve predicoes
    // A ideia é devolver um Evaluation, que possa, depois, ser tratada no
    // programa principal
    /**
     * Função para testar o modelo; 
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
     * Função que mostra a ajuda da ferramenta
     */
    static void showHelp(){
        System.out.println("comandos disponiveis:\n");
        System.out.println("training <algoritmo>  -> para treinar o algoritmos");
        System.out.println("test <algoritmo>  -> para testar os algoritmos");
        System.out.println("\n\n");
        System.out.println("Algoritmos disponiveis: \n");
        System.out.println("clonalg => Algoritmo CLONALG");
        System.out.println("mlp => Algoritmo Back-Propagation Multi-Layer Perceptron");
        System.out.println("\n");
    }
    
    /**
     * Função que prepara o dataset para a geração dos modelos
     * @param seed
     * @return 
     */
    public static Instances[] preparaDataset (int numAmostras,int seed){
        
        Instances dataset,dadosTreino,dadosTeste;
        
        Instances[] instances;
        
        System.out.println("=> Abre ficheiro");
        // Abre o ficheiro de treino
        dataset=abreDataset(ficheiro1);
        
        System.out.println("=> Trata Classe");
        // Trata a classe
        dataset=trataClasse(dataset);
        
        dataset=trataMissingValues(dataset);
        
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
     * Função para gerar/criar os modelos CLONALG
     * Guarda os modelos em ficheiro para uso posterior
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
     * Testa os modelos CLONALG, tendo como base os dados de treino e teste fornecidos
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
     * Função que gera o modelo ensemble doo CLONALG, combinando os 3 modelos individuais
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
     * Função que testa os modelos CLONALG utilizando um ficheiro de dados completo (ficheiro2)
     * @param seed 
     */
    public static void zeroDayCLONALG(int seed){
        
        Vote ensemble;
        // Abre segundo ficheiro de dados e usa-o  como dataset de testes
         
        Instances dataset;
        ArrayList predictions;
        
        System.out.println("=> Abre ficheiro de dados");
        // Abre o ficheiro de treino
        dataset=abreDataset(ficheiro2);
        
        estatisticaClasse(dataset);
        
        System.out.println("=> Trata Classe");
        // Trata a classe
        dataset=trataClasse(dataset);
        
        estatisticaClasse(dataset);
        
        dataset=trataMissingValues(dataset); 
        
        // Para efeitos de teste da ferramenta, vai buscar apenas 100000 amostras
        // No teste final, comentar as duas linhas seguintes
        // System.out.println("=> Faz o Resample");
        // dataset=divideDataset(dataset,100000,seed);
        
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
     * Gera a ROC Curve das predições
     * Código extraído de https://waikato.github.io/weka-wiki/generating_roc_curve/
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
        
        // Variáveis para opções dos modelos CLONALG
        //String[] optCLONALG1,optCLONALG2,optCLONALG3;
        
        
        
        
        
        /**
         * Global seed value
         */
        int globalSeed=1;
        
        
        // Trata argumentos da linha de comandos
        if (args.length<=1){
            showHelp();
            System.exit(0);
        }
        
        switch (args[0]){
            case "training":
                switch (args[1]){
                    case "clonalg":
                        // Gera CLONALG
                        System.out.println("Gerando os modelos CLONALG");
                        instances=preparaDataset(200000,globalSeed);
                
                        dadosTreino=instances[0];
                        // dadosTeste=instances[1]; // Not needed
                
                        geraModelosClonalg(dadosTreino,globalSeed);
                                
                        System.exit(0);
                        
                        break;
                    case "mlp":
                        // Gera BackMLP
                        System.out.println("Gerando os modelos BackMLP");
                        instances=preparaDataset(200000,globalSeed);
                
                        dadosTreino=instances[0];
                        // dadosTeste=instances[1]; // Not needed
                
                        geraModelosBackMLP(dadosTreino,globalSeed);
                                
                        System.exit(0);
                        break;
                }
                break;
            case "test":
                switch (args[1]){
                    case "clonalg":
                        // Testa CLONALG
                        System.out.println("Testando os modelos CLONALG");
                                 
                        instances=preparaDataset(200000,globalSeed); 
                        dadosTreino=instances[0];
                        dadosTeste=instances[1];
                
                        testaModelosClonalg(dadosTreino,dadosTeste,globalSeed);
                
                        zeroDayCLONALG(globalSeed);
                
                        input = new Scanner(System.in);
                        System.out.print("Press Enter to quit...");
                        input.nextLine();
                
                        System.exit(0);
                        break;
                    case "mlp":
                        // Testa BackMLP
                        System.out.println("Testando os modelos BackMLP");
                                 
                        instances=preparaDataset(200000,globalSeed); 
                        dadosTreino=instances[0];
                        dadosTeste=instances[1];
                
                        testaModelosBackMLP(dadosTreino,dadosTeste,globalSeed);
                
                        zeroDayBackMLP(globalSeed);
                
                        input = new Scanner(System.in);
                        System.out.print("Press Enter to quit...");
                        input.nextLine();
                
                        System.exit(0);
                        break;
                }
                break;
            case "ensemble":
                switch (args[1]){
                    case "clonalg":
                        
                        break;
                    case "mlp":
                        
                        break;
                    case "total":
                        
                        break;
                }
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
