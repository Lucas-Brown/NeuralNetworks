package genetics;

import java.util.Arrays;

import fullyConnectedNetwork.NetworkTools;
import parser.Attribute;
import parser.Node;
import parser.Parser;
import parser.ParserTools;
import trainSet.TrainSet;

public class GeneticMemoryNetwork extends GeneticNetwork {

	private double memory[][];
    private int memoryLength, memoryWidth;

    public GeneticMemoryNetwork(int ActivationFunction, double multiplier, int[] NETWORK_LAYER_SIZES) {
		super(ActivationFunction, multiplier, NETWORK_LAYER_SIZES);
	}

    public GeneticMemoryNetwork(int ActivationFunction, int memoryLength, int memoryWidth, int... NETWORK_LAYER_SIZES) {
        super(ActivationFunction, NETWORK_LAYER_SIZES);
        this.memoryLength = memoryLength;
        this.memoryWidth = memoryWidth;
        this.memory = new double[this.memoryLength][this.memoryWidth];
        for (double[] mem : this.memory) {
            Arrays.fill(mem, 1);
        }
    }

    public GeneticMemoryNetwork(int ActivationFunction, int memoryLength, int memoryWidth, double multiplier, int... NETWORK_LAYER_SIZES) {
        super(ActivationFunction, multiplier, NETWORK_LAYER_SIZES);
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
        
        double[] returned = new double[this.output.length - this.memoryWidth]; 
        double[] memorySet = new double[this.memoryWidth];
        System.arraycopy(this.output[this.NETWORK_SIZE - 1], 0, returned, 0, returned.length);
        System.arraycopy(this.output[this.NETWORK_SIZE - 1], returned.length, memorySet, 0, memorySet.length);
        this.push(this.memory, memorySet);
        return returned;
    }
 

    private void push(double[][] arr, double[] value) {
        for (int i = arr.length - 1; i > 1; i--) {
            arr[i] = arr[i - 1];
        }
        arr[0] = value;
    }

    public static int[] adjustLayers(int memoryLength,int memoryWidth, int... layers) {
        if (memoryLength < 0) {
            return layers;
        } else {
            for (int layer = 1; layer < layers.length; layer++) {
                layers[layer] += memoryWidth;
            }
            return layers;
        }
    }
    
	public void saveNetwork(String fileName) throws Exception {
        Parser p = new Parser();
        p.create(fileName);
        Node root = p.getContent();
        Node netw = new Node("Network");
        Node ly = new Node("Layers");
        netw.addAttribute(new Attribute("Activation Function", Integer.toString(this.ACTIVATION_FUNCTION)));
        netw.addAttribute(new Attribute("Multiplier", Double.toString(this.multiplier)));
        netw.addAttribute(new Attribute("fitness", Double.toString(this.fitness)));
        netw.addAttribute(new Attribute("length", Double.toString(this.memoryLength)));
        netw.addAttribute(new Attribute("width", Double.toString(this.memoryWidth)));
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
        p.close();
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
        GeneticMemoryNetwork ne = new GeneticMemoryNetwork(af, length, width, Multiplyer, si);
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
