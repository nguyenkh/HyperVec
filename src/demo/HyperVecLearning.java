package demo;

import io.sentence.PlainSentenceInputStream;
import io.word.CombinedWordInputStream;
import io.word.PushBackWordStream;
import io.word.WordInputStream;
import io.sentence.SentenceInputStream;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import common.wordnet.LexicalHypernym;
import vocab.Vocab;
import word2vec.MultiThreadWord2Vec;
import word2vec.multitask.Hyper2Vec;



public class HyperVecLearning {
    public static void main(String[] args) throws IOException{
        
        
        MultiThreadWord2Vec word2vec = null;
        String configFile = args[0]; 
        int size = Integer.parseInt(args[1]);
        int window = Integer.parseInt(args[2]);
        
        W2vProperties properties = new W2vProperties(configFile);
        boolean softmax = Boolean.parseBoolean(properties.getProperty("HierarchialSoftmax"));
        int negativeSamples = Integer.parseInt(properties.getProperty("NegativeSampling"));
        double subSampling = Double.parseDouble(properties.getProperty("SubSampling"));
        String trainDirPath = properties.getProperty("TrainDir");
        String outputFile = properties.getProperty("WordVectorFile");
        String vocabFile = properties.getProperty("VocabFile");
        
        outputFile = outputFile.replaceAll(".bin", "_" + size + ".bin");
        
        File trainDir = new File(trainDirPath);
        File[] trainFiles = trainDir.listFiles();
        System.out.println("Starting training using dir " + trainDirPath);
        System.out.println("Output file: " + outputFile);

        boolean learnVocab = !(new File(vocabFile)).exists();
        Vocab vocab = new Vocab(Integer.parseInt(properties.getProperty("MinFrequency")));
        if (!learnVocab)
            vocab.loadVocab(vocabFile);// ,minFrequency);
        else {
            ArrayList<WordInputStream> wordStreamList = new ArrayList<>();
            for (File trainFile: trainFiles) {
                WordInputStream wordStream = new PushBackWordStream(trainFile.getAbsolutePath(), 200);
                wordStreamList.add(wordStream);
            }
          
            CombinedWordInputStream wordStream = new CombinedWordInputStream(wordStreamList);
            vocab.learnVocabFromTrainStream(wordStream);
            // save vocabulary
            vocab.saveVocab(vocabFile);
        }
                
        word2vec = new Hyper2Vec(size, window, softmax, negativeSamples, subSampling);            
        Hyper2Vec hypervec = (Hyper2Vec) word2vec;
        
        LexicalHypernym hypeNoun = new LexicalHypernym(properties.getProperty("hypeNoun"),
                                                       properties.getProperty("cohypoNoun"),
                                                       properties.getProperty("featureNoun"), 
                                                       vocab);
        LexicalHypernym hypeVerb = new LexicalHypernym(properties.getProperty("hypeVerb"),
                                                       properties.getProperty("cohypoVerb"),
                                                       properties.getProperty("featureVerb"), 
                                                       vocab);
        hypervec.setLexicalHypeNoun(hypeNoun);
        outputFile = outputFile.replaceAll(".bin", "_HypeNoun.bin");
        hypervec.setLexicalHypeVerb(hypeVerb);
        outputFile = outputFile.replaceAll(".bin", "_HypeVerb.bin");
            

        word2vec.setVocab(vocab);
        word2vec.initNetwork();

        System.out.println("Start training");
        try {
            ArrayList<SentenceInputStream> inputStreams = new ArrayList<SentenceInputStream>();
            for (File trainFile: trainFiles) {
                SentenceInputStream sentenceInputStream = new PlainSentenceInputStream(
                    new PushBackWordStream(trainFile.getAbsolutePath(), 200));
                inputStreams.add(sentenceInputStream);
            }
            
            word2vec.trainModel(inputStreams); 
            word2vec.saveVector(outputFile, true);
            
            System.out.println("The vocab size: " + vocab.getVocabSize() + " words");
        } catch (IOException e) {
            System.exit(1);
        }

    }
}
