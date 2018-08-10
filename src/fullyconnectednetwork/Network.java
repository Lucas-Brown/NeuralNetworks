package fullyconnectednetwork;

import parser.Attribute;
import parser.Node;
import parser.Parser;
import parser.ParserTools;
import TrainSet.TrainSet;

import java.util.Arrays;

public class Network {

    public static final double LEARNING_RATE = 0.001;

    public static final int ZERO_OR_ONE = 0;
    public static final int NEGATIVE_ONE_OR_ONE = 1;
    public static final int ZERO_TO_ONE = 2;
    public static final int NEGATIVE_ONE_TO_ONE = 3;
    public static final int ZERO_JUMP_X = 4;
    public static final int NEGATIVE_X_JUMP_X = 5;
    public static final int RECTIFIER = 6; // 0 - infinity

    public final int ACTIVATION_FUNCTION;
    public final double multiplier;

    private double[][] output;
    private double[][][] weights;
    private double[][] bias;

    private double[][] error_signal;
    private double[][] output_derivative;

    public final int[] NETWORK_LAYER_SIZES;
    public final int INPUT_SIZE;
    public final int OUTPUT_SIZE;
    public final int NETWORK_SIZE;

    public double[][][][] weightDataPoints;
    public double[][][] biasDataPoints;
    public double[][][][] weightDerivativePoints;
    public double[][][] biasDerivativePoints;
    public double[][][][] weightCoefficients;
    public double[][][] biasCoefficients;

    public Network(int ActivationFunction, int... NETWORK_LAYER_SIZES) {
        this.multiplier = 1;
        this.ACTIVATION_FUNCTION = ActivationFunction;
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
        this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];

        this.output = new double[NETWORK_SIZE][];
        this.weights = new double[NETWORK_SIZE][][];
        this.bias = new double[NETWORK_SIZE][];

        this.weightDataPoints = new double[this.NETWORK_SIZE][][][];
        this.biasDataPoints = new double[this.NETWORK_SIZE][][];

        this.error_signal = new double[NETWORK_SIZE][];
        this.output_derivative = new double[NETWORK_SIZE][];

        for (int i = 0; i < NETWORK_SIZE; i++) {
            this.output[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.error_signal[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.output_derivative[i] = new double[NETWORK_LAYER_SIZES[i]];

            this.biasDataPoints[i] = new double[this.NETWORK_LAYER_SIZES[i]][];

            this.bias[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], 0, .5);

            if (i > 0) {
                this.weightDataPoints[i] = new double[this.NETWORK_LAYER_SIZES[i]][this.NETWORK_LAYER_SIZES[i - 1]][];
                this.weights[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], NETWORK_LAYER_SIZES[i - 1], 0, .5);
            }
        }
    }

    public Network(int ActivationFunction, double multiplier, int... NETWORK_LAYER_SIZES) {
        this.multiplier = multiplier;
        this.ACTIVATION_FUNCTION = ActivationFunction;
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
        this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];

        this.output = new double[NETWORK_SIZE][];
        this.weights = new double[NETWORK_SIZE][][];
        this.bias = new double[NETWORK_SIZE][];

        this.weightDataPoints = new double[this.NETWORK_SIZE][][][];
        this.biasDataPoints = new double[this.NETWORK_SIZE][][];

        this.error_signal = new double[NETWORK_SIZE][];
        this.output_derivative = new double[NETWORK_SIZE][];

        for (int i = 0; i < NETWORK_SIZE; i++) {
            this.output[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.error_signal[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.output_derivative[i] = new double[NETWORK_LAYER_SIZES[i]];

            this.biasDataPoints[i] = new double[this.NETWORK_LAYER_SIZES[i]][];

            this.bias[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], -1, 1);

            if (i > 0) {
                this.weightDataPoints[i] = new double[this.NETWORK_LAYER_SIZES[i]][this.NETWORK_LAYER_SIZES[i - 1]][];
                this.weights[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], NETWORK_LAYER_SIZES[i - 1], -1, 1);
            }
        }
    }

    public double[] calculate(double... input) {
        if (input.length != this.INPUT_SIZE) {
            return null;
        }
        switch (this.ACTIVATION_FUNCTION) {
            case 0:
                this.unitStepLoops(input);
                break;
            case 1:
                this.signumLoops(input);
                break;
            case 2:
                this.sigmoidLoops(input);
                break;
            case 3:
                this.hyperbolicTangentLoops(input);
                break;
            case 4:
                this.jumpStepLoops(input);
                break;
            case 5:
                this.jumpSignumLoops(input);
                break;
            case 6:
                this.rectifierLoops(input);
                break;
        }
        return output[NETWORK_SIZE - 1];
    }

    private void unitStepLoops(double... input) {
        this.output[0] = input;
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {

                double sum = bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    sum += output[layer - 1][prevNeuron] * weights[layer][neuron][prevNeuron];
                }
                output[layer][neuron] = this.unitStep(sum);
                output_derivative[layer][neuron] = output[layer][neuron];
            }
        }
    }

    private void signumLoops(double... input) {
        this.output[0] = input;
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {

                double sum = bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    sum += output[layer - 1][prevNeuron] * weights[layer][neuron][prevNeuron];
                }
                output[layer][neuron] = this.signum(sum);
                output_derivative[layer][neuron] = output[layer][neuron];
            }
        }
    }

    private void sigmoidLoops(double... input) {
        this.output[0] = input;
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {

                double sum = bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    sum += output[layer - 1][prevNeuron] * weights[layer][neuron][prevNeuron];
                }
                output[layer][neuron] = this.sigmoid(sum);
                output_derivative[layer][neuron] = Math.exp(-sum / this.multiplier) / Math.pow((1 + Math.exp(-sum / this.multiplier)), 2);
            }
        }
    }

    private void hyperbolicTangentLoops(double... input) {
        this.output[0] = input;
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {

                double sum = bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    sum += output[layer - 1][prevNeuron] * weights[layer][neuron][prevNeuron];
                }
                output[layer][neuron] = this.hyperbolicTangent(sum);
                output_derivative[layer][neuron] = this.multiplier * ((Math.exp(2 * sum / this.multiplier) - 1) / (Math.exp(2 * sum / this.multiplier) + 1));
            }
        }
    }

    private void jumpStepLoops(double... input) {
        this.output[0] = input;
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {

                double sum = bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    sum += output[layer - 1][prevNeuron] * weights[layer][neuron][prevNeuron];
                }
                output[layer][neuron] = this.jumpStep(sum);
                output_derivative[layer][neuron] = output[layer][neuron];
            }
        }
    }

    private void jumpSignumLoops(double... input) {
        this.output[0] = input;
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {

                double sum = bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    sum += output[layer - 1][prevNeuron] * weights[layer][neuron][prevNeuron];
                }
                output[layer][neuron] = this.jumpSignum(sum);
                output_derivative[layer][neuron] = output[layer][neuron];
            }
        }
    }

    private void rectifierLoops(double... input) {
        this.output[0] = input;
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {

                double sum = bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    sum += output[layer - 1][prevNeuron] * weights[layer][neuron][prevNeuron];
                }
                output[layer][neuron] = this.rectifier(sum);
                output_derivative[layer][neuron] = output[layer][neuron];
            }
        }
    }

    public void train(TrainSet set, int loops, int batch_size) {
        if (set.INPUT_SIZE != INPUT_SIZE || set.OUTPUT_SIZE != OUTPUT_SIZE) {
            return;
        }
        for (int i = 0; i < loops; i++) {
            TrainSet batch = set.extractBatch(batch_size);
            double mse = MSE(batch);
            for (int b = 0; b < batch_size; b++) {
                this.train(batch.getInput(b), batch.getOutput(b), LEARNING_RATE * mse);
            }
            System.out.println(mse);
        }
    }

    public void train(TrainSet set, int loops, int batch_size, int saveInterval, String file) {
        if (set.INPUT_SIZE != INPUT_SIZE || set.OUTPUT_SIZE != OUTPUT_SIZE) {
            return;
        }
        for (int i = 0; i < loops; i++) {
            TrainSet batch = set.extractBatch(batch_size);
            for (int b = 0; b < batch_size; b++) {
                this.train(batch.getInput(b), batch.getOutput(b), LEARNING_RATE);
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
        if (input.length != INPUT_SIZE || target.length != OUTPUT_SIZE) {
            return;
        }
        calculate(input);
        backpropError(target);
        updateWeights(eta);
    }

    public double MSE(double[] input, double[] target) {
        if (input.length != INPUT_SIZE || target.length != OUTPUT_SIZE) {
            return 0;
        }
        this.calculate(input);
        double v = 0;
        for (int i = 0; i < target.length; i++) {
            v += (target[i] - output[NETWORK_SIZE - 1][i]) * (target[i] - output[NETWORK_SIZE - 1][i]);
        }
        return v / (2d * target.length);
    }

    public double MSE(TrainSet set) {
        double v = 0;
        for (int i = 0; i < set.size(); i++) {
            v += MSE(set.getInput(i), set.getOutput(i));
        }
        return v / set.size();
    }

    public void backpropError(double[] target) {
        for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[NETWORK_SIZE - 1]; neuron++) {
            error_signal[NETWORK_SIZE - 1][neuron] = (output[NETWORK_SIZE - 1][neuron] - target[neuron])
                    * output_derivative[NETWORK_SIZE - 1][neuron];
        }
        for (int layer = NETWORK_SIZE - 2; layer > 0; layer--) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                double sum = 0;
                for (int nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZES[layer + 1]; nextNeuron++) {
                    sum += weights[layer + 1][nextNeuron][neuron] * error_signal[layer + 1][nextNeuron];
                }
                this.error_signal[layer][neuron] = sum * output_derivative[layer][neuron];
            }
        }
    }

    public void updateWeights(double eta) {
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {

                double delta = -eta * error_signal[layer][neuron];
                bias[layer][neuron] += delta;

                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    weights[layer][neuron][prevNeuron] += delta * output[layer - 1][prevNeuron];
                }
            }
        }
    }

    private double sigmoid(double x) {// 0 - 1
        return this.multiplier / (1 + Math.exp(-x / this.multiplier));
    }

    public double hyperbolicTangent(double x) { // -1 - 1
        return this.multiplier * ((Math.exp(x / this.multiplier) - Math.exp(-x / this.multiplier)) / (Math.exp(x / this.multiplier) + Math.exp(-x / this.multiplier)));
    }

    private double unitStep(double x) {// 1 or 0
        if (x > this.multiplier / 2) {
            return 1;
        } else if (x < this.multiplier / 2) {
            return 0;
        } else {
            return 0.5;
        }
    }

    private double signum(double x) {//values -1 or 1
        if (x > this.multiplier / 2) {
            return 1;
        } else if (x < this.multiplier / 2) {
            return -1;
        } else {
            return 0;
        }
    }

    private double jumpStep(double x) {
        if (x > this.multiplier) {
            return this.multiplier;
        } else if (x < 0) {
            return 0;
        } else {
            return Math.round(x);
        }
    }

    private double jumpSignum(double x) {
        if (x > this.multiplier) {
            return this.multiplier;
        } else if (x < -this.multiplier) {
            return -this.multiplier;
        } else {
            return Math.round(x);
        }
    }

    private double rectifier(double x) {
        return Math.log(1 + Math.exp(x));
    }

    public static void main(String[] args) {
        NN network = new NN(Network.ZERO_TO_ONE, 2, 3, 3, 1);

        Network.addPerfectExampleData(network);

        //network.addData(new double[]{0,0}, new double[]{0});
        network.train(10000);
        for (int i = 0; i < network.set.size(); i++) {
            System.out.println(Arrays.toString(network.set.getInput(i)) + " >--< " + Arrays.toString(network.calculate(network.set.getInput(i))) + 
                    ", should be: " + Arrays.toString(network.set.getOutput(i)));
        }
    }

    private static void addPerfectExampleData(NN network) {
        for(int degree = 5; degree <= 180; degree += 5){
            network.addData(new double[]{degree, 0}, new double[]{0});
        }
        for(int degree = 5; degree <= 180; degree += 5){
            for(double speed = .1; speed <= 1; speed += 0.1){
                network.addData(new double[]{degree, speed}, new double[]{(degree - degree / (90d * speed)) / 180});
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

    public void saveNetwork(String fileName) throws Exception {
        Parser p = new Parser();
        p.create(fileName);
        Node root = p.getContent();
        Node netw = new Node("Network");
        Node ly = new Node("Layers");
        netw.addAttribute(new Attribute("Activation Function", Integer.toString(this.ACTIVATION_FUNCTION)));
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

                w.addAttribute("" + we, Arrays.toString(weights[layer][we]));
            }
        }
        p.close();
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
}
