package genetics;

import java.util.Arrays;

import trainSet.TrainSet;

public class GeneticMemoryNetwork extends GeneticNetwork {

    public GeneticMemoryNetwork(int ActivationFunction, double multiplier, int[] NETWORK_LAYER_SIZES) {
		super(ActivationFunction, multiplier, NETWORK_LAYER_SIZES);
		// TODO Auto-generated constructor stub
	}

	private double memory[][];
    private int memoryLength, memoryWidth;

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
        for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) {

                double sum = this.bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < this.NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    sum += this.output[layer - 1][prevNeuron] * this.weights[layer][neuron][prevNeuron];
                }
                if (neuron >= this.NETWORK_LAYER_SIZES[layer] - this.memoryWidth + 1) {
                    if (sum < 0 || sum > this.memoryLength) { // default to most recent 
                        sum = 0;
                    }
                    sum = this.memory[(int) sum][this.NETWORK_LAYER_SIZES[layer] - neuron]; // the sum is the decider of what value it returns
                }
                switch (this.ACTIVATION_FUNCTION) {
                    case 0:
                    	this.output[layer][neuron] = this.unitStep(sum);
                    	this.output_derivative[layer][neuron] = this.output[layer][neuron];
                        break;
                    case 1:
                    	this.output[layer][neuron] = this.signum(sum);
                    	this.output_derivative[layer][neuron] = this.output[layer][neuron];
                        break;
                    case 2:
                    	this.output[layer][neuron] = this.sigmoid(sum);
                    	this.output_derivative[layer][neuron] = (this.multiplier * Math.exp(-sum)) / Math.pow(1 + Math.exp(-sum), 2);
                        break;
                    case 3:
                    	this.output[layer][neuron] = this.hyperbolicTangent(sum);
                    	this.output_derivative[layer][neuron] = (this.multiplier * Math.exp(-sum)) / (1 + Math.pow(Math.exp(-sum), 2));
                        break;
                    case 4:
                    	this.output[layer][neuron] = this.jumpStep(sum);
                    	this.output_derivative[layer][neuron] = this.output[layer][neuron];
                        break;
                    case 5:
                    	this.output[layer][neuron] = this.jumpSignum(sum);
                    	this.output_derivative[layer][neuron] = output[layer][neuron];
                        break;
                    case 6:
                    	this.output[layer][neuron] = this.rectifier(sum);
                    	this.output_derivative[layer][neuron] = this.output[layer][neuron];
                        break;
                }
            }
        }
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
}
