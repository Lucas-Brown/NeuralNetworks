package genetics;

import java.util.Arrays;

import fullyConnectedNetwork.ActivationFunction;
import fullyConnectedNetwork.NetworkTools;
import parser.Attribute;
import parser.Node;
import parser.Parser;
import parser.ParserTools;
import trainSet.TrainSet;

public class SelfAdjustingNetwork extends GeneticNetwork{ // uses the output of a few neurons as the bias and additional multiplier of extra hidden layer neurons

	protected int adjustingNeurons, finalLayerAdjustingNeurons;
	protected double[][] neuronMultiplier; 
	
	public SelfAdjustingNetwork(int ActivationFunction, int adjustingNeurons, double multiplier, int... NETWORK_LAYER_SIZES) {
		super(ActivationFunction, multiplier, SelfAdjustingNetwork.adjustLayers(adjustingNeurons, NETWORK_LAYER_SIZES));
		this.adjustingNeurons = adjustingNeurons;
		this.finalLayerAdjustingNeurons = 2 * this.adjustingNeurons * (this.NETWORK_LAYER_SIZES.length - 2);
		this.neuronMultiplier = NetworkTools.createRandomArray(this.NETWORK_SIZE - 1, this.adjustingNeurons, -1, 1);
	} 
	
	public SelfAdjustingNetwork(int ActivationFunction, int adjustingNeurons, int... NETWORK_LAYER_SIZES) {
		super(ActivationFunction, SelfAdjustingNetwork.adjustLayers(adjustingNeurons, NETWORK_LAYER_SIZES));
		this.adjustingNeurons = adjustingNeurons;
		this.finalLayerAdjustingNeurons = 2 * this.adjustingNeurons * (this.NETWORK_LAYER_SIZES.length - 2);
		this.neuronMultiplier = NetworkTools.createRandomArray(this.NETWORK_SIZE - 1, this.adjustingNeurons, -1, 1);
	} 
	
    @Override
    public void train(TrainSet set, int loops, int batch_size, int saveInterval, String file) {
        System.err.print("Cannot Train Self Adjusting Networks");
    }

    @Override
    public void train(TrainSet set, int loops, int batch_size) {
        System.err.print("Cannot Train Self Adjusting Networks");
    }

    @Override
    public void train(double[] input, double[] target, double eta) {
        System.err.print("Cannot Train Self Adjusting Networks");
    }
    
    @Override
    public double[] calculate(double... input) {
        if (input.length != this.INPUT_SIZE) {
            return null;
        }
        this.output[0] = input;
        this.loops(this.ACTIVATION_FUNCTION);
        
        NetworkTools.multiplyArray(this.output[this.NETWORK_SIZE - 1], this.multiplier);
        
        int finalNeuronNum = this.OUTPUT_SIZE - this.finalLayerAdjustingNeurons; //start at the first adjusting neuron
        for(int layer = 1; layer < this.NETWORK_SIZE - 2; layer++) {
        	for(int neuron = this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons; 
        	neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) {
        		this.bias[layer][neuron] = this.output[this.NETWORK_SIZE - 1][finalNeuronNum++];
        		this.neuronMultiplier[layer][neuron] = this.output[this.NETWORK_SIZE - 1][finalNeuronNum++];
        	}
        }
        
        double[] out = new double[this.OUTPUT_SIZE - this.finalLayerAdjustingNeurons];
        System.arraycopy(this.output[this.NETWORK_SIZE - 1], 0, out, 0, this.OUTPUT_SIZE - this.finalLayerAdjustingNeurons);
        
        return out;
    }
    
    @Override
    protected void loops(ActivationFunction AF) {
    	for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) {
                double sum = this.bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < this.NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    sum += this.output[layer - 1][prevNeuron] * this.weights[layer][neuron][prevNeuron];
                }
                
                if(layer == this.OUTPUT_SIZE - 1) {
                	if(neuron >= this.NETWORK_LAYER_SIZES[layer] - this.finalLayerAdjustingNeurons) {
                		sum *= this.neuronMultiplier[layer - 1][neuron - (this.NETWORK_LAYER_SIZES[layer] - this.finalLayerAdjustingNeurons)];
                		this.output[layer][neuron] = sum; // the final layer should not be subject to any activation function
                		this.output_derivative[layer][neuron] = 1;
                	}else {
                   		this.output[layer][neuron] = AF.activator(sum);
                   		this.output_derivative[layer][neuron] = Math.exp(-sum) / Math.pow((1 + Math.exp(-sum)), 2);
                	}
                }else {
                	if(neuron >= this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons) {
                		sum *= this.neuronMultiplier[layer - 1][neuron - (this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons)];
                	}
               		this.output[layer][neuron] = AF.activator(sum);
               		this.output_derivative[layer][neuron] = Math.exp(-sum) / Math.pow((1 + Math.exp(-sum)), 2);
                }
            }
        }
    }
    
	public static void breed (SelfAdjustingNetwork parentA, SelfAdjustingNetwork parentB, SelfAdjustingNetwork child, double MUTATION_RATE) {
        double percentWeight = (parentA.fitness * 0.5) / parentB.fitness; // weigh the randomizer towards the better parent

        for (int i = 0; i < child.neuronMultiplier.length; i++) {
            for (int j = 1; j < child.neuronMultiplier[i].length; j++) {
            	double parentVal = 0;
            	if(Math.random() < percentWeight) {
            		parentVal = parentA.neuronMultiplier[i][j];
            	} else {
            		parentVal = parentB.neuronMultiplier[i][j];
        		}
            	parentVal += (Math.random() * 2.0 - 1) * MUTATION_RATE;
            	child.neuronMultiplier[i][j] = parentVal;
            }
        }
	}
    
    static int[] adjustLayers(int adjuster, int... networkLayers) {
        if (adjuster <= 0) {
            return networkLayers;
        } else {
            for (int layer = 1; layer < networkLayers.length; layer++) {
            	if(layer == networkLayers.length - 1) { //last layer
            		networkLayers[layer] += 2 * adjuster * (networkLayers.length - 2); // an output neuron for every bias and weight for each additional neuron added to the hidden layers
            	}else {
            		networkLayers[layer] += adjuster;
            	}
            }
            return networkLayers;
        }
    }
    
    static int[] reverseAdjustLayers(int adjuster, int... networkLayers) {
        if (adjuster <= 0) {
            return networkLayers;
        } else {
            for (int layer = 1; layer < networkLayers.length; layer++) {
            	if(layer == networkLayers.length - 1) { //last layer
            		networkLayers[layer] -= 2 * adjuster * (networkLayers.length - 2); // an output neuron for every bias and weight for each additional neuron added to the hidden layers
            	}else {
            		networkLayers[layer] -= adjuster;
            	}
            }
            return networkLayers;
        }
    }

    @Override
	public void saveNetwork(String fileName) {
        Parser p = new Parser();
        p.create(fileName);
        Node root = p.getContent();
        Node netw = new Node("SelfAdjustingNetwork");
        Node ly = new Node("Layers");
        netw.addAttribute(new Attribute("Activation Function", Integer.toString(this.activationFunctionToInt())));
        netw.addAttribute(new Attribute("Adjustment", Integer.toString(this.adjustingNeurons)));
        netw.addAttribute(new Attribute("Multiplier", Double.toString(this.multiplier)));
        netw.addAttribute(new Attribute("fitness", Double.toString(this.fitness)));
        netw.addAttribute(new Attribute("sizes", Arrays.toString(this.NETWORK_LAYER_SIZES)));
        netw.addChild(ly);
        root.addChild(netw);
        for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {

            Node c = new Node("" + layer);
            ly.addChild(c);
            Node w = new Node("weights");
            Node b = new Node("biases");
            c.addChild(w);
            c.addChild(b);

            b.addAttribute("values", Arrays.toString(this.bias[layer]));

            for (int we = 0; we < this.weights[layer].length; we++) {

                w.addAttribute("" + we, Arrays.toString(this.weights[layer][we]));
            }
        }
        try {
			p.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
    }

    public static GeneticNetwork loadNetwork(String fileName) throws Exception {

        Parser p = new Parser();

        p.load(fileName);
        int af = Integer.parseInt(p.getValue(new String[]{"SelfAdjustingNetwork"}, "Activation Function"));
        int ad = Integer.parseInt(p.getValue(new String[]{"SelfAdjustingNetwork"}, "Adjustment"));
        double Multiplyer = Double.parseDouble(p.getValue(new String[]{"SelfAdjustingNetwork"}, "Multiplier"));
        double Fitness = Double.parseDouble(p.getValue(new String[]{"SelfAdjustingNetwork"}, "fitness"));
        String sizes = p.getValue(new String[]{"SelfAdjustingNetwork"}, "sizes");
        int[] si = ParserTools.parseIntArray(sizes);
        SelfAdjustingNetwork ne = new SelfAdjustingNetwork(af, ad, Multiplyer, reverseAdjustLayers(ad, si));
        ne.fitness = Fitness;
        
        for (int i = 1; i < ne.NETWORK_SIZE; i++) {
            String biases = p.getValue(new String[]{"SelfAdjustingNetwork", "Layers", new String(i + ""), "biases"}, "values");
            double[] bias = ParserTools.parseDoubleArray(biases);
            ne.bias[i] = bias;

            for (int n = 0; n < ne.NETWORK_LAYER_SIZES[i]; n++) {

                String current = p.getValue(new String[]{"SelfAdjustingNetwork", "Layers", new String(i + ""), "weights"}, "" + n);
                double[] val = ParserTools.parseDoubleArray(current);

                ne.weights[i][n] = val;
            }
        }
        p.close();
        return ne;

    }
}
