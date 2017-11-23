package word2vec.multitask;

import java.util.Set;

import org.ejml.simple.SimpleMatrix;

import word2vec.MultiThreadWord2Vec;
import common.SimpleMatrixUtils;
import common.wordnet.LexicalHypernym;

public class Hyper2Vec extends MultiThreadWord2Vec{
    
    protected LexicalHypernym hypeNoun;
    protected LexicalHypernym hypeVerb;
    protected double graded = 0.05;
    
    public Hyper2Vec(int projectionLayerSize, int windowSize,
            boolean hierarchicalSoftmax, int negativeSamples, double subSample) {
        super(projectionLayerSize, windowSize, hierarchicalSoftmax,
                negativeSamples, subSample);
    }
    
    public void setLexicalHypeNoun(LexicalHypernym hypeNoun) {
        this.hypeNoun = hypeNoun;
    }
    
    public void setLexicalHypeVerb(LexicalHypernym hypeVerb) {
        this.hypeVerb = hypeVerb;
    }
    
    public void trainSentence(int[] sentence) {
        // the parameter is a list of word's indices in the vocabulary
        // train with the sentence
        double[] a1 = new double[projectionLayerSize];
        double[] a1error = new double[projectionLayerSize];
        int sentenceLength = sentence.length;
        int iWordIndex = 0;
        // TODO: set the thing here
        
        boolean updateAtTheEnd=false;
        
        for (int wordPosition = 0; wordPosition < sentence.length; wordPosition++) {

            int wordIndex = sentence[wordPosition];

            // no way it will go here
            if (wordIndex == -1)
                continue;

            for (int i = 0; i < projectionLayerSize; i++) {
                a1[i] = 0;
                a1error[i] = 0;
            }

            // random actual window size
            int start = rand.nextInt(windowSize);

            //modality 1
            for (int i = start; i < windowSize * 2 + 1 - start; i++) {
                if (i != windowSize) {
                    int iPos = wordPosition - windowSize + i;
                    if (iPos < 0 || iPos >= sentenceLength)
                        continue;
                    iWordIndex = sentence[iPos];
                    if (iWordIndex == -1)
                        continue;
                                        
                    // NEGATIVE SAMPLING
                    for (int l = 0; l < negativeSamples + 1; l++) {
                        int target;
                        int label;

                        if (l == 0) {
                            target = iWordIndex;
                            label = 1;
                        } else {
                            target = unigram.randomWordIndex();
                            if (target == 0) {
                                target = rand.nextInt(vocab.getVocabSize() - 1) + 1;
                            }
                            if (target == iWordIndex)
                                continue;
                            label = 0;
                        }
                        double z2 = 0;
                        double gradient;
                        for (int j = 0; j < projectionLayerSize; j++) {
                            z2 += weights0[wordIndex][j]
                                    * negativeWeights1[target][j];
                        }
                        double a2 = sigmoidTable.getSigmoid(z2);
                        
                        gradient = (double) ((label - a2) * alpha);
                        for (int j = 0; j < projectionLayerSize; j++) {
                            a1error[j] += gradient
                                    * negativeWeights1[target][j];
                        }
                        for (int j = 0; j < projectionLayerSize; j++) {
                            negativeWeights1[target][j] += gradient
                                    * weights0[wordIndex][j];
                        }
                    }
                    // Learn weights input -> hidden
                    if (!updateAtTheEnd){
                        for (int j = 0; j < projectionLayerSize; j++) {
                            weights0[wordIndex][j] += a1error[j];
                            a1error[j] = 0;

                        }
                    }
                    ////////////////////////
                    // Learning Hypernymy //
                    ////////////////////////
                    
                    // Noun hypernyms
                    if (hypeNoun != null) {
                        if (hypeNoun.hasHypernyms(wordIndex) && hypeNoun.hasFeature(iWordIndex)) {
                            SimpleMatrix h1error = new SimpleMatrix(1, a1error.length);
                            SimpleMatrix wordVector = new SimpleMatrix(1, projectionLayerSize, true, weights0[wordIndex]);
                            SimpleMatrix contextVector = new SimpleMatrix(1, projectionLayerSize, true, weights0[iWordIndex]);
                            Set<Integer> features = hypeNoun.getFeatures(iWordIndex);
                            Set<Integer> hypernyms = hypeNoun.getHypernyms(wordIndex);
                            Integer count = 0;
                            double hypocos = SimpleMatrixUtils.cosine(contextVector, wordVector);
                            
                            for (Integer hypernymIndex: hypernyms) {
                                if (features.contains(hypernymIndex)) {
                                    SimpleMatrix hypernymVector = new SimpleMatrix(1, projectionLayerSize, true, weights0[hypernymIndex]);
                                    double hypecos = SimpleMatrixUtils.cosine(contextVector, hypernymVector);
                                    SimpleMatrix diff = new SimpleMatrix(1, a1error.length);
                                    double sim = hypocos - hypecos;
                                    if (sim >= graded) { 
                                        count += 1;
                                        diff = SimpleMatrixUtils.cosineDerivative(wordVector, hypernymVector);
                                        h1error = h1error.plus(diff);
                                    }
                                    else {
                                        diff = SimpleMatrixUtils.cosineDerivative(hypernymVector, wordVector);
                                        double[] errorArray = diff.getMatrix().data;
                                        for (int j = 0; j < projectionLayerSize; j++) {
                                            weights0[hypernymIndex][j] += errorArray[j]; 
                                        }
                                    }
                                }
                            }
                            if (count == 0) count = 1;
                            h1error = h1error.scale(1/count);
                            double[] errorArray = h1error.getMatrix().data;
                            // Learn weights input -> hidden
                            for (int j = 0; j < projectionLayerSize; j++) {
                                weights0[wordIndex][j] += errorArray[j]; 
                            }
                        }
                    }
                    // Verb hypernyms
                    if (hypeVerb != null) {
                        if (hypeVerb.hasHypernyms(wordIndex) && hypeVerb.hasFeature(iWordIndex)) {
                            SimpleMatrix h1error = new SimpleMatrix(1, a1error.length);
                            SimpleMatrix wordVector = new SimpleMatrix(1, projectionLayerSize, true, weights0[wordIndex]);
                            SimpleMatrix contextVector = new SimpleMatrix(1, projectionLayerSize, true, weights0[iWordIndex]);
                            Set<Integer> features = hypeVerb.getFeatures(iWordIndex);
                            Set<Integer> hypernyms = hypeVerb.getHypernyms(wordIndex);
                            Integer count = 0;
                            double hypocos = SimpleMatrixUtils.cosine(contextVector, wordVector);
                            for (Integer hypernymIndex: hypernyms) {
                                if (features.contains(hypernymIndex)) {
                                    SimpleMatrix hypernymVector = new SimpleMatrix(1, projectionLayerSize, true, weights0[hypernymIndex]);
                                    double hypecos = SimpleMatrixUtils.cosine(contextVector, hypernymVector);
                                    SimpleMatrix diff = new SimpleMatrix(1, a1error.length);
                                    double sim = hypocos - hypecos;
                                    if (sim >= graded) {
                                        count += 1;
                                        diff = SimpleMatrixUtils.cosineDerivative(wordVector, hypernymVector);
                                        h1error = h1error.plus(diff);
                                    }
                                    else {
                                        diff = SimpleMatrixUtils.cosineDerivative(hypernymVector, wordVector);
                                        double[] errorArray = diff.getMatrix().data;
                                        for (int j = 0; j < projectionLayerSize; j++) {
                                            weights0[hypernymIndex][j] += errorArray[j]; 
                                        }
                                    }
                                }
                            }
                            // Learn weights input -> hidden
                            if (count == 0) count = 1;
                            h1error = h1error.scale(1/count);
                            double[] errorArray = h1error.getMatrix().data;
                            // Learn weights input -> hidden
                            for (int j = 0; j < projectionLayerSize; j++) {
                                weights0[wordIndex][j] += errorArray[j]; 
                            }
                        }
                    }
                }
            }
        }
    }
}