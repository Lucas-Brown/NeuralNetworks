package genetics;

import java.util.Arrays;

import fullyConnectedNetwork.ActivationFunction;
import fullyConnectedNetwork.Network;
import fullyConnectedNetwork.NetworkTools;
import parser.Attribute;
import parser.Node;
import parser.Parser;
import parser.ParserTools;
import trainSet.TrainSet;

public class GeneticMemoryNetwork extends GeneticNetwork {

    // the length represents how long the memory bank is and the width determines
    // the number of memory neurons within each layer
    public int memoryLength, memoryWidth;
    private double memory[][];
    private IntArrList[][] reference;
    private double[][][] memoryError;
    private int trainingSetNum;

    public GeneticMemoryNetwork(ActivationFunction ActivationFunction, double multiplier, int... NETWORK_LAYER_SIZES) {
        super(ActivationFunction, multiplier, NETWORK_LAYER_SIZES);
    }

    public GeneticMemoryNetwork(ActivationFunction ActivationFunction, int memoryLength, int memoryWidth,
            int... NETWORK_LAYER_SIZES) {
        super(ActivationFunction, GeneticMemoryNetwork.adjustLayers(memoryLength, memoryWidth, NETWORK_LAYER_SIZES));
        this.memoryLength = memoryLength;
        this.memoryWidth = memoryWidth;
        this.memory = new double[this.memoryLength][this.memoryWidth];
        for (double[] mem : this.memory) {
            Arrays.fill(mem, 1);
        }
    }

    public GeneticMemoryNetwork(ActivationFunction ActivationFunction, int memoryLength, int memoryWidth,
            double multiplier, int... NETWORK_LAYER_SIZES) {
        super(ActivationFunction, multiplier,
                GeneticMemoryNetwork.adjustLayers(memoryLength, memoryWidth, NETWORK_LAYER_SIZES));
        this.memoryLength = memoryLength;
        this.memoryWidth = memoryWidth;
        this.memory = new double[this.memoryLength][this.memoryWidth];
        for (double[] mem : this.memory) {
            Arrays.fill(mem, 1);
        }
    }

    /*
     * Training the memory network required a strictly different set of rules, to
     * train a memory network we must take the full set of data, calculate the
     * outputs of the entire network, and work backwards to determine the error
     * since every neuron in the future can be affected by the past
     */
    @Override
    public void train(TrainSet set, int loops, int batch_size) {
        if (set.size() != batch_size) { // We do not want to have a random assortment of data
            return;
        }
        double[][][] outputs;
        for (int i = 0; i < loops; i++) {
            TrainSet batch = set.extractBatch(batch_size);
            outputs = new double[batch.size()][][];
            this.reference = new IntArrList[batch.size()][this.NETWORK_SIZE - 2]; //we do not include the input or the output layer
            this.memoryError = new double[batch.size()][][];
            for(int j = 0; j < this.memoryError.length; j++){
                this.memoryError[j] = this.memory.clone();
            }
            for (this.trainingSetNum = 0; this.trainingSetNum < batch.size(); this.trainingSetNum++) {
                this.calculate(batch.getInput(this.trainingSetNum));
                outputs[this.trainingSetNum] = this.output.clone(); // clone to ensure the entire array isn't referenced to
                                                               // one array
            }
            for(this.trainingSetNum = batch.size() - 1; this.trainingSetNum >= 0; this.trainingSetNum--){
                this.backpropError(batch.getOutput(this.trainingSetNum));
                this.output = outputs[this.trainingSetNum];
                this.error_signal = this.memoryError[this.trainingSetNum];
                this.updateWeights(Network.LEARNING_RATE);
            }
            System.out.println(this.MSE(batch));
        }
    }

    @Override
    public void train(double[] input, double[] target, double eta) {
        System.err.print("Cannot train memory networks for a single set of data");
    }

    @Override
    public double MSE(double[] input, double[] target) {
        if (input.length != this.INPUT_SIZE || target.length != this.OUTPUT_SIZE - this.memoryWidth) {
            System.err.println("MSE Error! " + input.length + ": " + this.INPUT_SIZE + ". " + target.length + ": "
                    + (this.OUTPUT_SIZE - 1));
            return 0;
        }
        calculate(input);
        double v = 0;
        for (int i = 0; i < target.length; i++) {
            v += (target[i] - this.output[this.NETWORK_SIZE - 1][i])
                    * (target[i] - this.output[this.NETWORK_SIZE - 1][i]);
        }
        return v / (2d * target.length);
    }

    // we can't call calculate() again without breaking the data flow.
    public double trainingMSE(double[] input, double[] target) {
        if (input.length != this.INPUT_SIZE || target.length != this.OUTPUT_SIZE - this.memoryWidth) {
            System.err.println("MSE Error! " + input.length + ": " + this.INPUT_SIZE + ". " + target.length + ": "
                    + (this.OUTPUT_SIZE - 1));
            return 0;
        }
        double v = 0;
        for (int i = 0; i < target.length; i++) {
            v += (target[i] - this.output[this.NETWORK_SIZE - 1][i])
                    * (target[i] - this.output[this.NETWORK_SIZE - 1][i]);
        }
        return v / (2d * target.length);
    }

    @Override
    public double[] calculate(double... input) {
        if (input.length != this.INPUT_SIZE) {
            return null;
        }
        this.output[0] = input;
        this.loops(this.ACTIVATION_FUNCTION);

        NetworkTools.multiplyArray(this.output[this.NETWORK_SIZE - 1], this.multiplier);

        double[] returned = new double[this.OUTPUT_SIZE - this.memoryWidth];
        double[] memorySet = new double[this.memoryWidth];
        System.arraycopy(this.output[this.NETWORK_SIZE - 1], 0, returned, 0, returned.length);
        System.arraycopy(this.output[this.NETWORK_SIZE - 1], returned.length, memorySet, 0, memorySet.length);
        NetworkTools.push(this.memory, memorySet);
        return returned;
    }

    /*
     * Calculating the backprop error will also need to record the error accumulated
     * from future memory neurons
     */
    @Override
    public void backpropError(double[] target) {
        // create the error signal of the final layer based uppon the target
        for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[this.NETWORK_SIZE - 1]; neuron++) {
            this.error_signal[this.NETWORK_SIZE
                    - 1][neuron] = ((this.output[this.NETWORK_SIZE - 1][neuron] - target[neuron])
                            * this.output_derivative[this.NETWORK_SIZE - 1][neuron]);
            if(this.trainingSetNum != this.memoryError.length && this.OUTPUT_SIZE - this.memoryWidth > 0){// if it's not the final set and it is a memory neuron
               for(int ref = 0; ref < this.reference[this.trainingSetNum][this.OUTPUT_SIZE - this.memoryWidth - 1].size(); ref++){
                    int[] pair = this.reference[this.trainingSetNum][this.OUTPUT_SIZE - this.memoryWidth - 1].get(ref);
                    this.error_signal[this.NETWORK_SIZE - 1][neuron] += this.memoryError[pair[0]][pair[1]][this.memoryWidth - neuron - 1];
                }
            }
        }
        /*
         * Use the error of the final layer to recursively trace through each layer to
         * calculate the error signal of each neuron
         */
        for (int layer = this.NETWORK_SIZE - 2; layer > 0; layer--) {
            for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) {
                double sum = 0;
                for (int nextNeuron = 0; nextNeuron < this.NETWORK_LAYER_SIZES[layer + 1]; nextNeuron++) {
                    sum += this.weights[layer + 1][nextNeuron][neuron] * this.error_signal[layer + 1][nextNeuron];
                }
                this.error_signal[layer][neuron] = sum * this.output_derivative[layer][neuron];
            }
        }
        this.memoryError[this.trainingSetNum] = this.error_signal.clone();
    }

    public void loops(ActivationFunction AF) {
        for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) {

                double sum = this.bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < this.NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    sum += this.output[layer - 1][prevNeuron] * this.weights[layer][neuron][prevNeuron];
                }
                if (neuron >= super.NETWORK_LAYER_SIZES[layer] - this.memoryWidth) {
                    if (sum < 0 || sum > this.memoryLength) { // default to no value
                        sum = 0;
                    } else {
                        sum = this.memory[(int) sum][super.NETWORK_LAYER_SIZES[layer] - 1 - neuron]; // the sum is the
                                                                                                     // decider of what
                                                                                                     // value the memory
                                                                                                     // returns
                        if (trainingSetNum != 0) {
                            reference[(int) (this.trainingSetNum - sum - 1)][layer].add(
                                    new int[] { this.trainingSetNum, super.NETWORK_LAYER_SIZES[layer] - 1 - neuron });
                            ;
                        }
                    }
                }
                this.output[layer][neuron] = AF.activator(sum);
                this.output_derivative[layer][neuron] = AF.derivative(sum);
            }
        }
    }

    public static int[] adjustLayers(int memoryLength, int memoryWidth, int... networkLayers) {
        if (memoryLength <= 0 || memoryWidth <= 0) {
            return networkLayers;
        } else {
            for (int layer = 1; layer < networkLayers.length; layer++) {
                networkLayers[layer] += memoryWidth;
            }
            return networkLayers;
        }
    }

    public static int[] reverseAdjustLayers(int memoryLength, int memoryWidth, int... networkLayers) {
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
        Node netw = new Node("GeneticMemoryNetwork");
        Node ly = new Node("Layers");
        netw.addAttribute(
                new Attribute("Activation Function", Integer.toString(this.ACTIVATION_FUNCTION.activationNum)));
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

    public static GeneticMemoryNetwork loadNetwork(String fileName) throws Exception {

        Parser p = new Parser();

        p.load(fileName);
        int af = Integer.parseInt(p.getValue(new String[] { "GeneticMemoryNetwork" }, "Activation Function"));
        double Multiplyer = Double.parseDouble(p.getValue(new String[] { "GeneticMemoryNetwork" }, "Multiplier"));
        double Fitness = Double.parseDouble(p.getValue(new String[] { "GeneticMemoryNetwork" }, "fitness"));
        int length = Integer.parseInt(p.getValue(new String[] { "GeneticMemoryNetwork" }, "length"));
        int width = Integer.parseInt(p.getValue(new String[] { "GeneticMemoryNetwork" }, "width"));
        String sizes = p.getValue(new String[] { "GeneticMemoryNetwork" }, "sizes");
        int[] si = ParserTools.parseIntArray(sizes);
        GeneticMemoryNetwork ne = new GeneticMemoryNetwork(ActivationFunction.intToActivationFunction(af), length,
                width, Multiplyer, GeneticMemoryNetwork.reverseAdjustLayers(length, width, si));
        ne.fitness = Fitness;

        for (int i = 1; i < ne.NETWORK_SIZE; i++) {
            String biases = p.getValue(new String[] { "GeneticMemoryNetwork", "Layers", new String(i + ""), "biases" },
                    "values");
            double[] bias = ParserTools.parseDoubleArray(biases);
            ne.bias[i] = bias;

            for (int n = 0; n < ne.NETWORK_LAYER_SIZES[i]; n++) {

                String current = p.getValue(
                        new String[] { "GeneticMemoryNetwork", "Layers", new String(i + ""), "weights" }, "" + n);
                double[] val = ParserTools.parseDoubleArray(current);

                ne.weights[i][n] = val;
            }
        }
        p.close();
        return ne;

    }

}
