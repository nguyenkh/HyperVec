package common;

import space.SemanticSpace;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

import common.IOUtils;

public class MeanAveragePrecision {
    String[][] wordPairs;
    double[] golds;
    
    public MeanAveragePrecision(String dataset) {
        readDataset(dataset);
    }
    
    public MeanAveragePrecision(String[][] wordPairs, double[] golds) {
        this.wordPairs = wordPairs;
        this.golds = golds;
    }
    
    public void readDataset(String dataset) {
        ArrayList<String> data = IOUtils.readFile(dataset);
        golds = new double[data.size()];
        wordPairs = new String[data.size()][2];
        for (int i = 0; i < data.size(); i++) {
            String dataPiece = data.get(i);
            String elements[] = dataPiece.split("\t");
            wordPairs[i][0] = elements[0];
            wordPairs[i][1] = elements[1];
            golds[i] = Double.parseDouble(elements[2]);
            //golds[i] = Double.parseDouble(elements[3]);
        }
    }
    
    public double evaluateMAP(SemanticSpace space) {
        final double[] predicts = new double[golds.length];
        for (int i = 0; i < golds.length; i++) {
            predicts[i] = space.getSim(wordPairs[i][0], wordPairs[i][1])
                          * space.getDirection(wordPairs[i][0], wordPairs[i][1]);
        }
        Integer[] idxs = new Integer[golds.length];
        for(int i = 0; i < golds.length; i++) idxs[i] = i;
        Arrays.sort(idxs, new Comparator<Integer>(){
            public int compare(Integer o1, Integer o2){
                return Double.compare(predicts[o2], predicts[o1]);
            }
        });
        double[] sorted_preds = new double[golds.length];
        for(int i = 0; i < golds.length; i++) sorted_preds[i] = golds[idxs[i]];
        
        double map = computeMAP(sorted_preds);
        return map;
    }
    
    public double computeMAP(double[] sorted_preds) {
        double ap = 0.0;
        double retrievedCounter = 0;
        double relevantCounter = 0;

        for (int i = 0; i < sorted_preds.length; i++) {
            retrievedCounter++;
            if (sorted_preds[i] == 1.0) {
                relevantCounter++;
                ap += relevantCounter / retrievedCounter;
            }
        }
        ap /= relevantCounter;
        return ap;
    }
    
    
}