package genetics;

import java.util.Arrays;

import fullyConnectedNetwork.NetworkTools;
import trainSet.TrainSet;

public class SelfAdjustingNetwork extends GeneticNetwork{ // uses the output of a few neurons as the bias and additional multiplier of extra hidden layer neurons

	int adjustingNeurons, finalLayerAdjustingNeurons;
	double[][] neuronMultiplier; 
	
	public SelfAdjustingNetwork(int ActivationFunction, int adjustingNeurons, double multiplier, int[] NETWORK_LAYER_SIZES) {
		super(ActivationFunction, multiplier, SelfAdjustingNetwork.adjustLayers(adjustingNeurons, NETWORK_LAYER_SIZES));
		this.adjustingNeurons = adjustingNeurons;
		this.finalLayerAdjustingNeurons = 2 * this.adjustingNeurons * (this.NETWORK_LAYER_SIZES.length - 2);
		this.neuronMultiplier = NetworkTools.createRandomArray(this.NETWORK_SIZE - 1, this.adjustingNeurons, -1, 1);
	} 
	
	public SelfAdjustingNetwork(int ActivationFunction, int adjustingNeurons, int[] NETWORK_LAYER_SIZES) {
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
    protected void unitStepLoops() {
        for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) {

                double sum = this.bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < this.NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    sum += this.output[layer - 1][prevNeuron] * this.weights[layer][neuron][prevNeuron];
                }
                this.output[layer][neuron] = this.unitStep(sum);
                this.output_derivative[layer][neuron] = this.output[layer][neuron];
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
                this.output[layer][neuron] = this.signum(sum);
                this.output_derivative[layer][neuron] = this.output[layer][neuron];
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
                if(neuron >= this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons) {
                	sum *= this.neuronMultiplier[layer - 1][neuron - (this.NETWORK_LAYER_SIZES[layer] - this.adjustingNeurons)];
                }
                if(layer == this.OUTPUT_SIZE - 1 && neuron >= this.NETWORK_LAYER_SIZES[layer] - this.finalLayerAdjustingNeurons) {
                	this.output[layer][neuron] = sum; // the final layer should not be subject to any activation function
                    this.output_derivative[layer][neuron] = 1;
                }else {
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
                this.output[layer][neuron] = this.hyperbolicTangent(sum);
                this.output_derivative[layer][neuron] = 4.0 / Math.pow((Math.exp(sum) + Math.exp(-sum)), 2);
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
                this.output[layer][neuron] = this.jumpStep(sum);
                this.output_derivative[layer][neuron] = this.output[layer][neuron];
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
                this.output[layer][neuron] = this.jumpSignum(sum);
                this.output_derivative[layer][neuron] = this.output[layer][neuron];
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
                this.output[layer][neuron] = this.rectifier(sum);
                this.output_derivative[layer][neuron] = this.output[layer][neuron];
            }
        }
    }

    
    private static int[] adjustLayers(int adjuster, int... networkLayers) {
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
}
