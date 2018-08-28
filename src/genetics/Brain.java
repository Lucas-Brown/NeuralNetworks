package genetics;

import java.util.Arrays;

import fullyConnectedNetwork.NetworkTools;
import parser.Attribute;
import parser.Node;
import parser.Parser;
import parser.ParserTools;
import trainSet.TrainSet;

public class Brain extends SelfAdjustingNetwork{ // this class is a compound of both GeneticMemoryNetwork and SelfAdjustingnetwork.
	
	//the order of neurons in the network is:
	// 1) regular
	// 2) memory
	// 3) adjusting
	
    private int memoryLength, memoryWidth;
	private double memory[][];
	
	public Brain(int ActivationFunction, int adjustingNeurons, int memoryLength, int memoryWidth, double multiplier, int... NETWORK_LAYER_SIZES) {
		super(ActivationFunction, adjustingNeurons, multiplier, GeneticMemoryNetwork.adjustLayers(memoryLength, memoryWidth, NETWORK_LAYER_SIZES));
        this.memoryLength = memoryLength;
        this.memoryWidth = memoryWidth;
        this.memory = new double[this.memoryLength][this.memoryWidth];
        for (double[] mem : this.memory) {
            Arrays.fill(mem, 1);
        }
	}
	
	public Brain(int ActivationFunction, int adjustingNeurons, int memoryLength, int memoryWidth, int... NETWORK_LAYER_SIZES) {
		super(ActivationFunction, adjustingNeurons, GeneticMemoryNetwork.adjustLayers(memoryLength, memoryWidth, NETWORK_LAYER_SIZES));
        this.memoryLength = memoryLength;
        this.memoryWidth = memoryWidth;
        this.memory = new double[this.memoryLength][this.memoryWidth];
        for (double[] mem : this.memory) {
            Arrays.fill(mem, 1);
        }
	}
	
    @Override
    public void train(TrainSet set, int loops, int batch_size, int saveInterval, String file) {
        System.err.print("Cannot Train Brain");
    }

    @Override
    public void train(TrainSet set, int loops, int batch_size) {
        System.err.print("Cannot Train Brain");
    }

    @Override
    public void train(double[] input, double[] target, double eta) {
        System.err.print("Cannot Train Brain");
    }
    
    @Override
    public double[] calculate(double... input) {
        if (input.length != this.INPUT_SIZE) {
            return null;
        }
        this.output[0] = input;
        switch (this.ACTIVATION_FUNCTION) {
            case 0:
                this.unitStepLoops();
                break;
            case 1:
                this.signumLoops();
                break;
            case 2:
                this.sigmoidLoops();
                break;
            case 3:
                this.hyperbolicTangentLoops();
                break;
            case 4:
                this.jumpStepLoops();
                break;
            case 5:
                this.jumpSignumLoops();
                break;
            case 6:
                this.rectifierLoops();
                break;
        }
        
        NetworkTools.multiplyArray(this.output[this.NETWORK_SIZE - 1], this.multiplier);
        
        double[] returned = new double[this.OUTPUT_SIZE - this.finalLayerAdjustingNeurons - this.memoryWidth]; 
        double[] memorySet = new double[this.memoryWidth];
        System.arraycopy(this.output[this.NETWORK_SIZE - 1], 0, returned, 0, returned.length);
        System.arraycopy(this.output[this.NETWORK_SIZE - 1], returned.length, memorySet, 0, memorySet.length);
        NetworkTools.push(this.memory, memorySet); // add the memory neuron array to the memory bank
        
        int finalNeuronNum = this.OUTPUT_SIZE - this.finalLayerAdjustingNeurons; //start at the first adjusting neuron
        for(int layer = 1; layer < this.NETWORK_SIZE - 2; layer++) {
        	for(int neuron = this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons; 
        	neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) {
        		this.bias[layer][neuron] = this.output[this.NETWORK_SIZE - 1][finalNeuronNum++];
        		this.neuronMultiplier[layer][neuron] = this.output[this.NETWORK_SIZE - 1][finalNeuronNum++];
        	}
        }
        
        return returned;
    }
    
    @Override
    protected void unitStepLoops() {
        for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) {

                double sum = this.bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < this.NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    sum += this.output[layer - 1][prevNeuron] * this.weights[layer][neuron][prevNeuron];
                }
                if(neuron >= this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons) {
                	sum *= this.neuronMultiplier[layer - 1][neuron - (this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons)];
                }
                if(layer == this.OUTPUT_SIZE - 1 && neuron >= this.NETWORK_LAYER_SIZES[layer] - this.finalLayerAdjustingNeurons) {
                	this.output[layer][neuron] = sum; // the final layer should not be subject to any activation function
                    this.output_derivative[layer][neuron] = 1;
                }else {
                	this.output[layer][neuron] = this.unitStep(sum);
                	this.output_derivative[layer][neuron] = this.output[layer][neuron];
                }
            }
        }
    }

    @Override
    protected void signumLoops() {
        for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) {

                double sum = this.bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < this.NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    sum += this.output[layer - 1][prevNeuron] * this.weights[layer][neuron][prevNeuron];
                }
                if(neuron >= this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons) {
                	sum *= this.neuronMultiplier[layer - 1][neuron - (this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons)];
                }
                if(layer == this.OUTPUT_SIZE - 1 && neuron >= this.NETWORK_LAYER_SIZES[layer] - this.finalLayerAdjustingNeurons) {
                	this.output[layer][neuron] = sum; // the final layer should not be subject to any activation function
                    this.output_derivative[layer][neuron] = 1;
                }else {
                	this.output[layer][neuron] = this.signum(sum);
                	this.output_derivative[layer][neuron] = this.output[layer][neuron];
                }
            }
        }
    }

    @Override
    protected void sigmoidLoops() {
        for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) {
                double sum = this.bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < this.NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    sum += this.output[layer - 1][prevNeuron] * this.weights[layer][neuron][prevNeuron];
                }

                if(layer == this.OUTPUT_SIZE - 1) {
                	if (neuron >= super.NETWORK_LAYER_SIZES[layer] - this.memoryWidth - this.finalLayerAdjustingNeurons && neuron < this.NETWORK_LAYER_SIZES[layer] - this.finalLayerAdjustingNeurons) {
                        if (sum < 0 || sum > this.memoryLength) { // default to no value
                            sum = 0;
                        }else{
                            sum = this.memory[(int) sum][super.NETWORK_LAYER_SIZES[layer] - 1 - neuron - this.finalLayerAdjustingNeurons]; // the sum is the decider of what value the memory returns
                        }
                   		this.output[layer][neuron] = this.sigmoid(sum);
                   		this.output_derivative[layer][neuron] = Math.exp(-sum) / Math.pow((1 + Math.exp(-sum)), 2);
                    }else if(neuron >= this.NETWORK_LAYER_SIZES[layer] - this.finalLayerAdjustingNeurons) {
                		sum *= this.neuronMultiplier[layer - 1][neuron - (this.NETWORK_LAYER_SIZES[layer] - this.finalLayerAdjustingNeurons)];
                		this.output[layer][neuron] = sum; // the final layer should not be subject to any activation function
                		this.output_derivative[layer][neuron] = 1;
                	}else {
                   		this.output[layer][neuron] = this.sigmoid(sum);
                   		this.output_derivative[layer][neuron] = Math.exp(-sum) / Math.pow((1 + Math.exp(-sum)), 2);
                	}
                	
                }else {
                	if (neuron >= super.NETWORK_LAYER_SIZES[layer] - this.memoryWidth - this.adjustingNeurons && neuron < this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons) {
                        if (sum < 0 || sum > this.memoryLength) { // default to no value
                            sum = 0;
                        }else{
                            sum = this.memory[(int) sum][super.NETWORK_LAYER_SIZES[layer] - 1 - neuron - this.adjustingNeurons]; // the sum is the decider of what value the memory returns
                        }
                    }else if(neuron >= this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons) {
                		sum *= this.neuronMultiplier[layer - 1][neuron - (this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons)];
                	}
               		this.output[layer][neuron] = this.sigmoid(sum);
               		this.output_derivative[layer][neuron] = Math.exp(-sum) / Math.pow((1 + Math.exp(-sum)), 2);
                }
            }
        }
    }

    @Override
    protected void hyperbolicTangentLoops() {
        for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) {

                double sum = this.bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < this.NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    sum += this.output[layer - 1][prevNeuron] * this.weights[layer][neuron][prevNeuron];
                }
                if(neuron >= this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons) {
                	sum *= this.neuronMultiplier[layer - 1][neuron - (this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons)];
                }
                if(layer == this.OUTPUT_SIZE - 1 && neuron >= this.NETWORK_LAYER_SIZES[layer] - this.finalLayerAdjustingNeurons) {
                	this.output[layer][neuron] = sum; // the final layer should not be subject to any activation function
                    this.output_derivative[layer][neuron] = 1;
                }else {
                	this.output[layer][neuron] = this.hyperbolicTangent(sum);
                	this.output_derivative[layer][neuron] = 4.0 / Math.pow((Math.exp(sum) + Math.exp(-sum)), 2);
                }
            }
        }
    }

    @Override
    protected void jumpStepLoops() {
        for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) {

                double sum = this.bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < this.NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    sum += this.output[layer - 1][prevNeuron] * this.weights[layer][neuron][prevNeuron];
                }
                if(neuron >= this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons) {
                	sum *= this.neuronMultiplier[layer - 1][neuron - (this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons)];
                }
                if(layer == this.OUTPUT_SIZE - 1 && neuron >= this.NETWORK_LAYER_SIZES[layer] - this.finalLayerAdjustingNeurons) {
                	this.output[layer][neuron] = sum; // the final layer should not be subject to any activation function
                    this.output_derivative[layer][neuron] = 1;
                }else {
                	this.output[layer][neuron] = this.jumpStep(sum);
                	this.output_derivative[layer][neuron] = this.output[layer][neuron];
                }
            }
        }
    }

    @Override
    protected void jumpSignumLoops() {
        for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) {

                double sum = this.bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < this.NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    sum += this.output[layer - 1][prevNeuron] * this.weights[layer][neuron][prevNeuron];
                }
                if(neuron >= this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons) {
                	sum *= this.neuronMultiplier[layer - 1][neuron - (this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons)];
                }
                if(layer == this.OUTPUT_SIZE - 1 && neuron >= this.NETWORK_LAYER_SIZES[layer] - this.finalLayerAdjustingNeurons) {
                	this.output[layer][neuron] = sum; // the final layer should not be subject to any activation function
                    this.output_derivative[layer][neuron] = 1;
                }else {
                	this.output[layer][neuron] = this.jumpSignum(sum);
                	this.output_derivative[layer][neuron] = this.output[layer][neuron];
                }
            }
        }
    }

    protected void rectifierLoops() {
        for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) {

                double sum = this.bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < this.NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    sum += this.output[layer - 1][prevNeuron] * this.weights[layer][neuron][prevNeuron];
                }
                if(neuron >= this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons) {
                	sum *= this.neuronMultiplier[layer - 1][neuron - (this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons)];
                }
                if(layer == this.OUTPUT_SIZE - 1 && neuron >= this.NETWORK_LAYER_SIZES[layer] - this.finalLayerAdjustingNeurons) {
                	this.output[layer][neuron] = sum; // the final layer should not be subject to any activation function
                    this.output_derivative[layer][neuron] = 1;
                }else {
                	this.output[layer][neuron] = this.rectifier(sum);
                	this.output_derivative[layer][neuron] = this.output[layer][neuron];
                }
            }
        }
    }
    
	public static void breed (Brain parentA, Brain parentB, Brain child, double MUTATION_RATE) {
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
    
    private static int[] adjustLayers(int adjuster, int memoryLength, int memoryWidth, int... networkLayers) {
        if (adjuster <= 0 && (memoryLength <= 0 || memoryWidth <= 0)) {
            return networkLayers;
        }else if(adjuster <= 0 && (memoryLength > 0 || memoryWidth > 0)) {
        	return GeneticMemoryNetwork.adjustLayers(memoryLength, memoryWidth, networkLayers);
        }else if(adjuster > 0 && (memoryLength <= 0 || memoryWidth <= 0)) {
        	return SelfAdjustingNetwork.adjustLayers(adjuster, networkLayers);
        }else {
            for (int layer = 1; layer < networkLayers.length; layer++) {
            	networkLayers[layer] += memoryWidth;
            	if(layer == networkLayers.length - 1) { //last layer
            		networkLayers[layer] += 2 * adjuster * (networkLayers.length - 2); // an output neuron for every bias and weight for each additional neuron added to the hidden layers
            	}else {
            		networkLayers[layer] += adjuster;
            	}
            }
            return networkLayers;
        }
    }
    
    static int[] reverseAdjustLayers(int adjuster, int memoryLength, int memoryWidth, int... networkLayers) {
        if (adjuster <= 0 && (memoryLength <= 0 || memoryWidth <= 0)) {
            return networkLayers;
        }else if(adjuster <= 0 && (memoryLength > 0 || memoryWidth > 0)) {
        	return GeneticMemoryNetwork.adjustLayers(memoryLength, memoryWidth, networkLayers);
        }else if(adjuster > 0 && (memoryLength <= 0 || memoryWidth <= 0)) {
        	return SelfAdjustingNetwork.adjustLayers(adjuster, networkLayers);
        }else {
            for (int layer = 1; layer < networkLayers.length; layer++) {
            	networkLayers[layer] -= memoryWidth;
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
        Node netw = new Node("Network");
        Node ly = new Node("Layers");
        netw.addAttribute(new Attribute("Activation Function", Integer.toString(this.ACTIVATION_FUNCTION)));
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
        int af = Integer.parseInt(p.getValue(new String[]{"Network"}, "Activation Function"));
        int ad = Integer.parseInt(p.getValue(new String[]{"Network"}, "Adjustment"));
        double Multiplyer = Double.parseDouble(p.getValue(new String[]{"Network"}, "Multiplier"));
        double Fitness = Double.parseDouble(p.getValue(new String[]{"Network"}, "fitness"));
        String sizes = p.getValue(new String[]{"Network"}, "sizes");
        int[] si = ParserTools.parseIntArray(sizes);
        SelfAdjustingNetwork ne = new SelfAdjustingNetwork(af, ad, Multiplyer, reverseAdjustLayers(ad, si));
        ne.fitness = Fitness;
        
        for (int i = 1; i < ne.NETWORK_SIZE; i++) {
            String biases = p.getValue(new String[]{"Network", "Layers", new String(i + ""), "biases"}, "values");
            double[] bias = ParserTools.parseDoubleArray(biases);
            ne.bias[i] = bias;

            for (int n = 0; n < ne.NETWORK_LAYER_SIZES[i]; n++) {

                String current = p.getValue(new String[]{"Network", "Layers", new String(i + ""), "weights"}, "" + n);
                double[] val = ParserTools.parseDoubleArray(current);

                ne.weights[i][n] = val;
            }
        }
        p.close();
        return ne;

    }
}