package com.tcl.weilong.mace;

import android.content.Context;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class DelOutput {
    private static final int RESULTS_TO_SHOW = 3;
    public List<String> loadLabelList(Context context) throws Exception {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(context.getAssets().open("labels.txt")));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }
    public PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(RESULTS_TO_SHOW, new Comparator<Map.Entry<String, Float>>() {
                @Override
                public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                    return (o1.getValue()).compareTo(o2.getValue());
                }
            });
    /** Prints top-K labels, to be shown in UI as the results. */
    public String printTopKLabels(List<String> labelList,float[] labelProbArrayFloat) {
        for (int i = 0; i < labelList.size(); ++i) {
            sortedLabels.add(
                    new AbstractMap.SimpleEntry<>(labelList.get(i), labelProbArrayFloat[i]));
            if (sortedLabels.size() > RESULTS_TO_SHOW) {
                sortedLabels.poll();
            }
        }
        String textToShow = "";
        final int size = sortedLabels.size();
        float value = 0;
        for (int i = 0; i < size; ++i) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            if (value < label.getValue()) {
                value = label.getValue();
                textToShow = label.getKey() + ":" + Float.toString(label.getValue());
            }
        }
        return textToShow;
    }

}
