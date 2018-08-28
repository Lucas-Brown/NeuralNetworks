package fullyConnectedNetwork;

import parser.Attribute;
import parser.Node;
import parser.Parser;
import parser.ParserTools;
import trainSet.TrainSet;

import java.util.Arrays;

import genetics.Brain;

public class Network {

    public static final double LEARNING_RATE = 0.001;

    public static final int ZERO_OR_ONE = 0;
    public static final int NEGATIVE_ONE_OR_ONE = 1;
    public static final int ZERO_TO_ONE = 2;
    public static final int NEGATIVE_ONE_TO_ONE = 3;
    public static final int ZERO_JUMP_X = 4;
    public static final int NEGATIVE_X_JUMP_X = 5;
    public static final int RECTIFIER = 6; // 0 - infinity

    public final ActivationFunction ACTIVATION_FUNCTION;
    public final double multiplier;

    public double[][] output;
    public double[][][] weights;
    public double[][] bias;

    private double[][] error_signal;
    protected double[][] output_derivative;

    public final int[] NETWORK_LAYER_SIZES;
    public final int INPUT_SIZE;
    public final int OUTPUT_SIZE;
    public final int NETWORK_SIZE;

    //public double[][][][] weightDataPoints;
    //public double[][][] biasDataPoints;
    //public double[][][][] weightDerivativePoints;
    //public double[][][] biasDerivativePoints;
    //public double[][][][] weightCoefficients;
    //public double[][][] biasCoefficients;

    public Network(int ActivationFunction, int... NETWORK_LAYER_SIZES) {
    	switch (ActivationFunction) {
        case 0:
        	this.ACTIVATION_FUNCTION = new UnitStep();
            break;
        case 1:
        	this.ACTIVATION_FUNCTION = new Signum();
            break;
        case 2:
        	this.ACTIVATION_FUNCTION = new Sigmoid();
            break;
        case 3:
        	this.ACTIVATION_FUNCTION = new HyperbolicTangent();
            break;
        case 4:
        	this.ACTIVATION_FUNCTION = new JumpStep();
            break;
        case 5:
        	this.ACTIVATION_FUNCTION = new JumpSignum();
            break;
        case 6:
        	this.ACTIVATION_FUNCTION = new Rectifier();
            break;
        default:
        	this.ACTIVATION_FUNCTION = null;
        	break;
    	}
    	
        this.multiplier = 1;
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
        this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];

        this.output = new double[NETWORK_SIZE][];
        this.weights = new double[NETWORK_SIZE][][];
        this.bias = new double[NETWORK_SIZE][];

        //this.weightDataPoints = new double[this.NETWORK_SIZE][][][];
        //this.biasDataPoints = new double[this.NETWORK_SIZE][][];

        this.error_signal = new double[NETWORK_SIZE][];
        this.output_derivative = new double[NETWORK_SIZE][];

        for (int i = 0; i < NETWORK_SIZE; i++) {
            this.output[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.error_signal[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.output_derivative[i] = new double[NETWORK_LAYER_SIZES[i]];

            //this.biasDataPoints[i] = new double[this.NETWORK_LAYER_SIZES[i]][];

            this.bias[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i],  -0.5, 0.7);

            if (i > 0) {
                //this.weightDataPoints[i] = new double[this.NETWORK_LAYER_SIZES[i]][this.NETWORK_LAYER_SIZES[i - 1]][];
                this.weights[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], NETWORK_LAYER_SIZES[i - 1], -0.5, 0.7);
            }
        }
    }

    public Network(int ActivationFunction, double multiplier, int... NETWORK_LAYER_SIZES) {
    	if(multiplier <= 0) {
    		System.out.println("multiplier cannot be less than or equal to zero");
    		this.multiplier = 1;
    	} else {
    		this.multiplier = multiplier;
    	}

    	switch (ActivationFunction) {
        case 0:
        	this.ACTIVATION_FUNCTION = new UnitStep();
            break;
        case 1:
        	this.ACTIVATION_FUNCTION = new Signum();
            break;
        case 2:
        	this.ACTIVATION_FUNCTION = new Sigmoid();
            break;
        case 3:
        	this.ACTIVATION_FUNCTION = new HyperbolicTangent();
            break;
        case 4:
        	this.ACTIVATION_FUNCTION = new JumpStep();
            break;
        case 5:
        	this.ACTIVATION_FUNCTION = new JumpSignum();
            break;
        case 6:
        	this.ACTIVATION_FUNCTION = new Rectifier();
            break;
        default:
        	this.ACTIVATION_FUNCTION = null;
        	break;
    	}
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
        this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];

        this.output = new double[NETWORK_SIZE][];
        this.weights = new double[NETWORK_SIZE][][];
        this.bias = new double[NETWORK_SIZE][];

        //this.weightDataPoints = new double[this.NETWORK_SIZE][][][];
        //this.biasDataPoints = new double[this.NETWORK_SIZE][][];

        this.error_signal = new double[NETWORK_SIZE][];
        this.output_derivative = new double[NETWORK_SIZE][];

        for (int i = 0; i < NETWORK_SIZE; i++) {
            this.output[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.error_signal[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.output_derivative[i] = new double[NETWORK_LAYER_SIZES[i]];

            //this.biasDataPoints[i] = new double[this.NETWORK_LAYER_SIZES[i]][];

            this.bias[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i],  -0.5, 0.7);

            if (i > 0) {
                //this.weightDataPoints[i] = new double[this.NETWORK_LAYER_SIZES[i]][this.NETWORK_LAYER_SIZES[i - 1]][];
                this.weights[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], NETWORK_LAYER_SIZES[i - 1], -0.5, 0.7);
            }
        }
    }

    public double[] calculate(double... input) {
        if (input.length != this.INPUT_SIZE) {
            return null;
        }
        this.output[0] = input;
        
        this.loops(this.ACTIVATION_FUNCTION);
        NetworkTools.multiplyArray(this.output[this.NETWORK_SIZE - 1], this.multiplier);
        
        return this.output[this.NETWORK_SIZE - 1];
    }
    
    protected void loops(ActivationFunction AF) {
        for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) {

                double sum = this.bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < this.NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    sum += this.output[layer - 1][prevNeuron] * this.weights[layer][neuron][prevNeuron];
                }
                this.output[layer][neuron] = AF.activator(sum);
                this.output_derivative[layer][neuron] = Math.exp(-sum) / Math.pow((1 + Math.exp(-sum)), 2);
            }
        }
    }

    public void train(TrainSet set, int loops, int batch_size) {
        if (set.INPUT_SIZE != this.INPUT_SIZE || set.OUTPUT_SIZE != this.OUTPUT_SIZE) {
            return;
        }
        for (int i = 0; i < loops; i++) {
            TrainSet batch = set.extractBatch(batch_size);
            double mse = MSE(batch);
            for (int b = 0; b < batch_size; b++) {
                this.train(batch.getInput(b), batch.getOutput(b), Network.LEARNING_RATE);
            }
            System.out.println(mse);
        }
    }

    public void train(TrainSet set, int loops, int batch_size, int saveInterval, String file) {
        if (set.INPUT_SIZE != this.INPUT_SIZE || set.OUTPUT_SIZE != this.OUTPUT_SIZE) {
            return;
        }
        for (int i = 0; i < loops; i++) {
            TrainSet batch = set.extractBatch(batch_size);
            for (int b = 0; b < batch_size; b++) {
                this.train(batch.getInput(b), batch.getOutput(b), Network.LEARNING_RATE);
            }
            System.out.println(MSE(batch));
            if (i % saveInterval == 0) {
                try {
                    saveNetwork(file);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
        try {
            saveNetwork(file);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void train(double[] input, double[] target, double eta) {
        if (input.length != this.INPUT_SIZE || target.length != this.OUTPUT_SIZE) {
            return;
        }
        calculate(input);
        backpropError(target);
        updateWeights(eta);
    }

    public double MSE(double[] input, double[] target) {
        if (input.length != this.INPUT_SIZE || target.length != this.OUTPUT_SIZE) {
            return 0;
        }
        this.calculate(input);
        double v = 0;
        for (int i = 0; i < target.length; i++) {
            v += (target[i] - this.output[this.NETWORK_SIZE - 1][i]) * (target[i] - this.output[NETWORK_SIZE - 1][i]);
        }
        return v / (2d * target.length * this.multiplier * this.multiplier);
    }

    public double MSE(TrainSet set) {
        double v = 0;
        for (int i = 0; i < set.size(); i++) {
            v += MSE(set.getInput(i), set.getOutput(i));
        }
        return v / set.size();
    }

    public void backpropError(double[] target) {
        for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[this.NETWORK_SIZE - 1]; neuron++) {
        	this.error_signal[this.NETWORK_SIZE - 1][neuron] = ((this.output[this.NETWORK_SIZE - 1][neuron] - target[neuron])
                    * this.output_derivative[this.NETWORK_SIZE - 1][neuron]) / this.multiplier;
        }
        for (int layer = this.NETWORK_SIZE - 2; layer > 0; layer--) {
            for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) {
                double sum = 0;
                for (int nextNeuron = 0; nextNeuron < this.NETWORK_LAYER_SIZES[layer + 1]; nextNeuron++) {
                    sum += this.weights[layer + 1][nextNeuron][neuron] * this.error_signal[layer + 1][nextNeuron];
                }
                this.error_signal[layer][neuron] = sum * this.output_derivative[layer][neuron];
            }
        }
    }

    public void updateWeights(double eta) {
        for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {

                double delta = -eta * this.error_signal[layer][neuron];
                this.bias[layer][neuron] += delta;

                for (int prevNeuron = 0; prevNeuron < this.NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                	this.weights[layer][neuron][prevNeuron] += delta * this.output[layer - 1][prevNeuron];
                }
            }
        }
    }

    public class Sigmoid extends ActivationFunction{

		@Override
		public double activator(double x) {
			return 1 / (1 + Math.exp(-x));
		}
    }
    
    public class HyperbolicTangent extends ActivationFunction{

		@Override
		public double activator(double x) {
			return Math.tanh(x);
		}
    }
    
    public class UnitStep extends ActivationFunction{

		@Override
		public double activator(double x) {
			if (x > 0.5) {
	            return 1;
	        } else {
	            return 0;
	        }
		}
    }

    public class Signum extends ActivationFunction{

		@Override
		public double activator(double x) {
			if (x >= 0) {
	            return 1;
	        } else {
	        	return -1;
	        }
		}
    }
    
    public class JumpStep extends ActivationFunction{

		@Override
		public double activator(double x) {
			if (x > 0.5) {
	            return 1;
	        } else {
	            return 0;
	        } 
		}
    }
    
    public class JumpSignum extends ActivationFunction{

		@Override
		public double activator(double x) {
			if (x > 1) {
	            return 1;
	        } else if (x < -1) {
	            return -1;
	        } else {
	            return Math.round(x);
	        }
		}
    }

    public class Rectifier extends ActivationFunction{

		@Override
		public double activator(double x) {
			return Math.log(1 + Math.exp(x));
		}
    }

    public static void main(String[] args) {
    	class smallGroup extends NetworkGroup{

			public smallGroup(Network[][] networks) {
				super(networks);
			}

			@Override
			public double[] calculate(double... input) {
				this.groupOutput = new double[this.group[1][0].OUTPUT_SIZE + this.group[1][1].OUTPUT_SIZE];
				
				double[] firstOutput = this.group[0][0].calculate(input);
				System.arraycopy(this.group[1][0].calculate(new double[] {firstOutput[0], firstOutput[1]}), 0, this.groupOutput, 0, this.group[1][0].OUTPUT_SIZE);
				System.arraycopy(this.group[1][1].calculate(new double[] {firstOutput[2], firstOutput[3], firstOutput[4]}), 0, this.groupOutput, this.group[1][0].OUTPUT_SIZE, this.group[1][1].OUTPUT_SIZE);
				return this.groupOutput;
			}
    		
    	}
    	
    	Brain brian = new Brain(Network.ZERO_TO_ONE, 2, 20, 3, 10.0, 10, 7, 5);
    	Network net1 = new Network(Network.ZERO_TO_ONE, 7.0, 2, 2, 2);
    	Network net2 = new Network(Network.ZERO_OR_ONE, 3, 2, 2); 
    	
    	smallGroup sg = new smallGroup(new Network[][] {{brian}, {net1, net2}});
    	System.out.println(Arrays.toString(sg.calculate(new double[] {
    			1, 2, 3, 4, 5, 6, 7, 8, 9, 10})));
    }

    private static void addPerfectExampleData(NN network) {
        for(int degree = 5; degree <= 180; degree += 5){
            network.addData(new double[]{degree, 0}, new double[]{0});
        }
        for(int degree = 5; degree <= 180; degree += 5){
            for(double speed = .1; speed <= 1; speed += 0.1){
                network.addData(new double[]{degree, speed}, new double[]{network.net.multiplier * ((degree - degree / (90d * speed)) / 180)});
            }
        }
    }
    
    private static void addExampleData(NN network) {
        network.addData(new double[]{0, 0}, new double[]{0});
        network.addData(new double[]{0, 0.2}, new double[]{0});
        network.addData(new double[]{0, 0.4}, new double[]{0});
        network.addData(new double[]{0, 0.6}, new double[]{0});
        network.addData(new double[]{0, 0.8}, new double[]{0});
        network.addData(new double[]{0, 1.0}, new double[]{0});

        network.addData(new double[]{10, 0.4}, new double[]{9});
        network.addData(new double[]{15, 0.4}, new double[]{14});
        network.addData(new double[]{20, 0.4}, new double[]{18});
        network.addData(new double[]{25, 0.4}, new double[]{23});
        network.addData(new double[]{30, 0.4}, new double[]{27});
        network.addData(new double[]{35, 0.4}, new double[]{32});
        network.addData(new double[]{40, 0.4}, new double[]{37});
        network.addData(new double[]{45, 0.4}, new double[]{42});
        network.addData(new double[]{50, 0.4}, new double[]{47});
        network.addData(new double[]{55, 0.4}, new double[]{52});
        network.addData(new double[]{60, 0.4}, new double[]{57});
        network.addData(new double[]{65, 0.4}, new double[]{62});
        network.addData(new double[]{70, 0.4}, new double[]{67});
        network.addData(new double[]{75, 0.4}, new double[]{71});
        network.addData(new double[]{80, 0.4}, new double[]{76});
        network.addData(new double[]{85, 0.4}, new double[]{81});
        network.addData(new double[]{90, 0.4}, new double[]{85});

        network.addData(new double[]{10, 0.6}, new double[]{9});
        network.addData(new double[]{15, 0.6}, new double[]{14});
        network.addData(new double[]{20, 0.6}, new double[]{17});
        network.addData(new double[]{25, 0.6}, new double[]{22});
        network.addData(new double[]{30, 0.6}, new double[]{27});
        network.addData(new double[]{35, 0.6}, new double[]{32});
        network.addData(new double[]{40, 0.6}, new double[]{37});
        network.addData(new double[]{45, 0.6}, new double[]{41});
        network.addData(new double[]{50, 0.6}, new double[]{46});
        network.addData(new double[]{55, 0.6}, new double[]{51});
        network.addData(new double[]{60, 0.6}, new double[]{56});
        network.addData(new double[]{65, 0.6}, new double[]{60});
        network.addData(new double[]{70, 0.6}, new double[]{65});
        network.addData(new double[]{75, 0.6}, new double[]{70});
        network.addData(new double[]{80, 0.6}, new double[]{75});
        network.addData(new double[]{85, 0.6}, new double[]{79});
        network.addData(new double[]{90, 0.6}, new double[]{84});

        network.addData(new double[]{10, 0.8}, new double[]{8});
        network.addData(new double[]{15, 0.8}, new double[]{13});
        network.addData(new double[]{20, 0.8}, new double[]{17});
        network.addData(new double[]{25, 0.8}, new double[]{22});
        network.addData(new double[]{30, 0.8}, new double[]{27});
        network.addData(new double[]{35, 0.8}, new double[]{32});
        network.addData(new double[]{40, 0.8}, new double[]{36});
        network.addData(new double[]{45, 0.8}, new double[]{41});
        network.addData(new double[]{50, 0.8}, new double[]{46});
        network.addData(new double[]{55, 0.8}, new double[]{50});
        network.addData(new double[]{60, 0.8}, new double[]{55});
        network.addData(new double[]{65, 0.8}, new double[]{60});
        network.addData(new double[]{70, 0.8}, new double[]{65});
        network.addData(new double[]{75, 0.8}, new double[]{69});
        network.addData(new double[]{80, 0.8}, new double[]{74});
        network.addData(new double[]{85, 0.8}, new double[]{79});
        network.addData(new double[]{90, 0.8}, new double[]{83});
    }

    public void saveNetwork(String fileName) {
        Parser p = new Parser();
        p.create(fileName);
        Node root = p.getContent();
        Node netw = new Node("Network");
        Node ly = new Node("Layers");
        netw.addAttribute(new Attribute("Activation Function", Integer.toString(this.activationFunctionToInt())));
        netw.addAttribute(new Attribute("Multiplier", Double.toString(this.multiplier)));
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

    public static Network loadNetwork(String fileName) throws Exception {

        Parser p = new Parser();

        p.load(fileName);
        int af = Integer.parseInt(p.getValue(new String[]{"Network"}, "Activation Function"));
        double Multiplyer = Double.parseDouble(p.getValue(new String[]{"Network"}, "Multiplier"));
        String sizes = p.getValue(new String[]{"Network"}, "sizes");
        int[] si = ParserTools.parseIntArray(sizes);
        Network ne = new Network(af, Multiplyer, si);

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
    
    public static Network copy(Network network) {
        Network coppied = new Network(network.activationFunctionToInt(), network.multiplier, network.NETWORK_LAYER_SIZES);
        for (int i = 0; i < network.bias.length; i++) {
            System.arraycopy(network.bias[i], 0, coppied.bias[i], 0, network.bias[i].length);
        }
        for (int i = 1; i < network.weights.length; i++) {
            for (int j = 0; j < network.weights[i].length; j++) {
                System.arraycopy(network.weights[i][j], 0, coppied.weights[i][j], 0, network.weights[i][j].length);
            }
        }
        return coppied;
    }
    
    public int activationFunctionToInt() {
    	if(this.ACTIVATION_FUNCTION.equals(new UnitStep())){
    		return 0;
    	}else if(this.ACTIVATION_FUNCTION.equals(new Signum())){
    		return 1;
    	}else if(this.ACTIVATION_FUNCTION.equals(new Sigmoid())){
    		return 2;
    	}else if(this.ACTIVATION_FUNCTION.equals(new HyperbolicTangent())){
    		return 3;
    	}else if(this.ACTIVATION_FUNCTION.equals(new JumpStep())){
    		return 4;
    	}else if(this.ACTIVATION_FUNCTION.equals(new JumpSignum())){
    		return 5;
    	}else if(this.ACTIVATION_FUNCTION.equals(new Rectifier())){
    		return 6;
    	}else{ 
    		return 7;
    	} 
    }
}
