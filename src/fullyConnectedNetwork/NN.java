package fullyConnectedNetwork;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;

import trainSet.TrainSet;

public class NN { // a simplification of the Network and TrainSet classes

    private Network net;
    private TrainSet set;

    public NN(ActivationFunction ActivationFunction, int... NETWORK_LAYER_SIZES) {
        this.net = new Network(ActivationFunction, NETWORK_LAYER_SIZES);
        this.set = new TrainSet(this.net.INPUT_SIZE, this.net.OUTPUT_SIZE);
    }
    
    public NN(ActivationFunction ActivationFunction, double multiplyer, int... NETWORK_LAYER_SIZES) {
        this.net = new Network(ActivationFunction, multiplyer, NETWORK_LAYER_SIZES);
        this.set = new TrainSet(this.net.INPUT_SIZE, this.net.OUTPUT_SIZE);
    }

    public NN(String netPath, String trainPath) {
        try {
            this.net = Network.loadNetwork(netPath);
            this.set = new TrainSet(trainPath);
        } catch (Exception ex) {
            Logger.getLogger(NN.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void train(int loops){
        net.train(this.set, loops, set.size());
    }
    
    public void trainAndSave(int loops, int saveInterval, String file){
        net.train(set, loops, set.size(), saveInterval, file);
    }
    
    public void jumpTrain(int loops, int jumpsNum, double endingLearningRate) { // for creating new networks based on the data set.
    	this.net = this.net.jumpTrain(this.net, this.set, loops, this.set.size(), jumpsNum, endingLearningRate);
    }
    
    public void addData(double[] input, double[] expected){
        this.set.addData(input, expected);
    }
    
    public void saveSet(String path){
        this.set.saveTrainSet(path);
    }
    
    public void saveNet(String path) throws Exception{
        this.net.saveNetwork(path);
    }
    
    public void saveNN(String netPath, String setPath) {
    	try {
			this.saveNet(netPath);
			this.saveSet(setPath);
		} catch (Exception e) {
			e.printStackTrace();
		}
    }
    
    public double[] calculate(double[] input){
        return this.net.calculate(input);
    }
    
    public static void addTrainDataToFile(double[] input, double[] expected, String file) {
    	String currentData = "";
    	
    	Scanner sc = null;
    	try {
    		System.out.println(new File("").getAbsolutePath());
    		sc = new Scanner(new File(file));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	while(sc.hasNext()) {
    		currentData += sc.nextLine();
    	}
    		
    	PrintWriter pw = null;
    	try {
			pw = new PrintWriter(new FileWriter(new File(file)));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	pw.print(currentData + "\n");
    	pw.println(Arrays.toString(input) + ", " + Arrays.toString(expected) + ", \n");
    }

    
}
