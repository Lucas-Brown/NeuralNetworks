package genetics;

import java.util.Arrays;

import fullyConnectedNetwork.ActivationFunction;
import fullyConnectedNetwork.NetworkTools;
import parser.Attribute;
import parser.Node;
import parser.Parser;
import parser.ParserTools;
import trainSet.TrainSet;

public class GeneticMemoryNetwork extends GeneticNetwork {

    private int memoryLength, memoryWidth;
	private double memory[][];
    //the length represents how long the memory bank is and the width determines the number of memory neurons within each layer

    public GeneticMemoryNetwork(int ActivationFunction, double multiplier, int... NETWORK_LAYER_SIZES) {
		super(ActivationFunction, multiplier, NETWORK_LAYER_SIZES);
	}

    public GeneticMemoryNetwork(int ActivationFunction, int memoryLength, int memoryWidth, int... NETWORK_LAYER_SIZES) {
        super(ActivationFunction, GeneticMemoryNetwork.adjustLayers(memoryLength, memoryWidth, NETWORK_LAYER_SIZES));
        this.memoryLength = memoryLength;
        this.memoryWidth = memoryWidth;
        this.memory = new double[this.memoryLength][this.memoryWidth];
        for (double[] mem : this.memory) {
            Arrays.fill(mem, 1);
        }
    }

    public GeneticMemoryNetwork(int ActivationFunction, int memoryLength, int memoryWidth, double multiplier, int... NETWORK_LAYER_SIZES) {
        super(ActivationFunction, multiplier, GeneticMemoryNetwork.adjustLayers(memoryLength, memoryWidth, NETWORK_LAYER_SIZES));
        this.memoryLength = memoryLength;
        this.memoryWidth = memoryWidth;
        this.memory = new double[this.memoryLength][this.memoryWidth];
        for (double[] mem : this.memory) {
            Arrays.fill(mem, 1);
        }
    }

    @Override
    public void train(TrainSet set, int loops, int batch_size, int saveInterval, String file) {
        System.err.print("Cannot Train Memory Networks");
    }

    @Override
    public void train(TrainSet set, int loops, int batch_size) {
        System.err.print("Cannot Train Memory Networks");
    }

    @Override
    public void train(double[] input, double[] target, double eta) {
        System.err.print("Cannot Train Memory Networks");
    }

    @Override
    public double MSE(double[] input, double[] target) {
        if (input.length != this.INPUT_SIZE || target.length != this.OUTPUT_SIZE - this.memoryWidth) {
            System.err.println("MSE Error! " + input.length + ": " + this.INPUT_SIZE + ". " + target.length + ": " + (this.OUTPUT_SIZE - 1));
            return 0;
        }
        calculate(input);
        double v = 0;
        for (int i = 0; i < target.length; i++) {
            v += (target[i] - this.output[this.NETWORK_SIZE - 1][i]) * (target[i] - this.output[this.NETWORK_SIZE - 1][i]);
        }
        return v / (2d * target.length);
    }

    @Override
    public double[] calculate(double... input) {
        if (input.length != this.INPUT_SIZE) {
            return null;
        }
        this.output[0] = input;
        ActivationFunction AF = null;
        switch (this.ACTIVATION_FUNCTION) {
            case 0:
            	AF = new UnitStep();
                break;
            case 1:
            	AF = new Signum();
                break;
            case 2:
                AF = new Sigmoid();
                break;
            case 3:
                AF = new HyperbolicTangent();
                break;
            case 4:
                AF = new JumpStep();
                break;
            case 5:
                AF = new JumpSignum();
                break;
            case 6:
                AF = new Rectifier();
                break;
        }
        
        for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) {

            	double sum = this.bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < this.NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    sum += this.output[layer - 1][prevNeuron] * this.weights[layer][neuron][prevNeuron];
                }
                if (neuron >= super.NETWORK_LAYER_SIZES[layer] - this.memoryWidth) {
                    if (sum < 0 || sum > this.memoryLength) { // default to no value
                        sum = 0;
                    }else{
                        sum = this.memory[(int) sum][super.NETWORK_LAYER_SIZES[layer] - 1 - neuron]; // the sum is the decider of what value the memory returns
                    }
                }
                this.output[layer][neuron] = AF.activator(sum);
                this.output_derivative[layer][neuron] = Math.exp(-sum) / Math.pow((1 + Math.exp(-sum)), 2);
            }
        }
        
        NetworkTools.multiplyArray(this.output[this.NETWORK_SIZE - 1], this.multiplier);
        
        double[] returned = new double[this.OUTPUT_SIZE - this.memoryWidth]; 
        double[] memorySet = new double[this.memoryWidth];
        System.arraycopy(this.output[this.NETWORK_SIZE - 1], 0, returned, 0, returned.length);
        System.arraycopy(this.output[this.NETWORK_SIZE - 1], returned.length, memorySet, 0, memorySet.length);
        NetworkTools.push(this.memory, memorySet);
        return returned;
    }
    
    static int[] adjustLayers(int memoryLength, int memoryWidth, int... networkLayers) {
        if (memoryLength <= 0 || memoryWidth <= 0) {
            return networkLayers;
        } else {
            for (int layer = 1; layer < networkLayers.length; layer++) {
            	networkLayers[layer] += memoryWidth;
            }
            return networkLayers;
        }
    }
    
    static int[] reverseAdjustLayers(int memoryLength, int memoryWidth, int... networkLayers) {
        if (memoryLength <= 0 || memoryWidth <= 0) {
            return networkLayers;
        } else {
            for (int layer = 1; layer < networkLayers.length; layer++) {
            	networkLayers[layer] -= memoryWidth;
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
        netw.addAttribute(new Attribute("Multiplier", Double.toString(this.multiplier)));
        netw.addAttribute(new Attribute("fitness", Double.toString(this.fitness)));
        netw.addAttribute(new Attribute("length", Integer.toString(this.memoryLength)));
        netw.addAttribute(new Attribute("width", Integer.toString(this.memoryWidth)));
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
        double Multiplyer = Double.parseDouble(p.getValue(new String[]{"Network"}, "Multiplier"));
        double Fitness = Double.parseDouble(p.getValue(new String[]{"Network"}, "fitness"));
        int length = Integer.parseInt(p.getValue(new String[] {"Network"}, "length"));
        int width = Integer.parseInt(p.getValue(new String[] {"Network"}, "width"));
        String sizes = p.getValue(new String[]{"Network"}, "sizes");
        int[] si = ParserTools.parseIntArray(sizes);
        GeneticMemoryNetwork ne = new GeneticMemoryNetwork(af, length, width, Multiplyer, GeneticMemoryNetwork.reverseAdjustLayers(length, width, si));
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
