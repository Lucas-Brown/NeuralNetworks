package trainSet;

import java.io.FileNotFoundException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.io.File;

import fullyConnectedNetwork.NetworkTools;
import parser.Attribute;
import parser.Node;
import parser.Parser;
import parser.ParserTools;

public class TrainSet {

    public final int INPUT_SIZE, OUTPUT_SIZE;

    //double[][] <- index1: 0 = input, 1 = output || index2: index of element
    private ArrayList<double[][]> data = new ArrayList<>();

    public TrainSet(int INPUT_SIZE, int OUTPUT_SIZE) {
        this.INPUT_SIZE = INPUT_SIZE;
        this.OUTPUT_SIZE = OUTPUT_SIZE;
    }
    
    public TrainSet(String path) throws Exception{
    	TrainSet loadedSet = TrainSet.loadTrainSet(path);
    	this.INPUT_SIZE = loadedSet.INPUT_SIZE;
    	this.OUTPUT_SIZE  = loadedSet.OUTPUT_SIZE;
    	this.data = loadedSet.data;
    }
    
    public static TrainSet loadTrainSet(String fileName) throws Exception {
        Parser p = new Parser();
        p.load(fileName);

        int in = Integer.parseInt(p.getValue(new String[]{"TrainSet"}, "Input Size"));
        int out = Integer.parseInt(p.getValue(new String[]{"TrainSet"}, "Output Size"));
        int size = Integer.parseInt(p.getValue(new String[]{"TrainSet"}, "Size"));
        TrainSet set = new TrainSet(in, out);
        

        for(int dataSet = 0; dataSet < size; dataSet++) {
        	double[] input = ParserTools.parseDoubleArray(p.getValue(new String[]{"TrainSet", "Data", "" + dataSet, "in"}, "input"));
        	double[] output = ParserTools.parseDoubleArray(p.getValue(new String[]{"TrainSet", "Data", "" + dataSet, "out"}, "output"));
        	set.addData(input, output);
        }
        
        p.close();
        return set;
    }
    
    public void saveTrainSet(String fileName){
        Parser p = new Parser();
        p.create(fileName);
        Node root = p.getContent();
        Node set = new Node("TrainSet");
        Node data = new Node("Data");
        set.addAttribute(new Attribute("Input Size", Integer.toString(this.INPUT_SIZE)));
        set.addAttribute(new Attribute("Output Size", Integer.toString(this.OUTPUT_SIZE)));
        set.addAttribute(new Attribute("Size", Integer.toString(this.data.size())));
        set.addChild(data);
        root.addChild(set);
        
        for(int dataSet = 0; dataSet < this.data.size(); dataSet++) {
            Node num = new Node("" + dataSet);
            data.addChild(num);

            Node i = new Node("in");
            Node o = new Node("out");
            num.addChild(i);
            num.addChild(o);
            
            i.addAttribute(new Attribute("input", Arrays.toString(this.data.get(dataSet)[0])));
            o.addAttribute(new Attribute("output", Arrays.toString(this.data.get(dataSet)[1])));
        }

        try {
			p.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
    }
    
    public void addData(double[] in, double[] expected) {
        if(in.length != INPUT_SIZE || expected.length != OUTPUT_SIZE) return;
        data.add(new double[][]{in, expected});
    }
    
    public void replaceData(int location, double[] in, double[] expected) {
        if(in.length != INPUT_SIZE || expected.length != OUTPUT_SIZE) return;
        this.data.set(location, new double[][]{in, expected});
    }

    public TrainSet extractBatch(int size) {
        if(size > 0 && size < this.size()) {
            TrainSet set = new TrainSet(INPUT_SIZE, OUTPUT_SIZE);
            Integer[] ids = NetworkTools.randomValues(0, this.size() - 1, size);
            for(Integer i:ids) {
                set.addData(this.getInput(i),this.getOutput(i));
            }
            return set;
        }else return this;
    }

    public static void main(String[] args) {
        TrainSet set = new TrainSet(3,2);

        for(int i = 0; i < 8; i++) {
            double[] a = new double[3];
            double[] b = new double[2];
            for(int k = 0; k < 3; k++) {
                a[k] = (double)((int)(Math.random() * 10)) / (double)10;
                if(k < 2) {
                    b[k] = (double)((int)(Math.random() * 10)) / (double)10;
                }
            }
            set.addData(a,b);
        }

        System.out.println(set);
        System.out.println(set.extractBatch(3));
    }

    public String toString() {
        String s = "TrainSet ["+INPUT_SIZE+ " ; "+OUTPUT_SIZE+"]\n";
        int index = 0;
        for(double[][] r:data) {
            s += index +":   "+Arrays.toString(r[0]) +"  >-||-<  "+Arrays.toString(r[1]) +"\n";
            index++;
        }
        return s;
    }

    public int size() {
        return data.size();
    }

    public double[] getInput(int index) {
        if(index >= 0 && index < size())
            return data.get(index)[0];
        else return null;
    }

    public double[] getOutput(int index) {
        if(index >= 0 && index < size())
            return data.get(index)[1];
        else return null;
    }

    public int getINPUT_SIZE() {
        return INPUT_SIZE;
    }

    public int getOUTPUT_SIZE() {
        return OUTPUT_SIZE;
    }
}
