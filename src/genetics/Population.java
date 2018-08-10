package genetics;

import fullyconnectednetwork.Network;
import java.io.File;
import java.math.BigInteger;
import java.util.Arrays;

public class Population {

    public GeneticNetwork[] population;
    //public final double[] target = new double[]{1, 1, 0, 0};
    public static final double MUTATION_RATE = 0.05;
    public final int ActivationFunction;

    public Population(int popSize, int af, int mutiplier, int... NETWORK_LAYER_SIZES) {
        this.population = new GeneticNetwork[popSize];
        this.ActivationFunction = af;
        for (int i = 0; i < this.population.length; i++) {
            this.population[i] = new GeneticNetwork(af, mutiplier, NETWORK_LAYER_SIZES); // automatically randomly generates new and unique networks
        }
    }

    public static void main(String[] args) {
        Population pop = new Population(75, Network.RECTIFIER, 10, 3, 3, 3, 1); // initialize population
        //for (int i = 0; i < 5; i++) { // load current top 5 networks
        //try {
        //pop.population[i] = Network.loadNetwork(new File("").getAbsolutePath() + "\\src\\saves\\top" + i + ".txt");
        //} catch (Exception ex) {
        //System.err.println(ex);
        //}
        //}
        while (true) { // repeat 
            pop.Fitness(); // evaluate fitness
            //pop.FitnessBase(); // compare to base
            System.out.println("Fitness complete");
            pop.nextGeneration(); // replace old generation with new one
            System.out.println("generation killed");
        }
    }

    public void Fitness() {
        for (GeneticNetwork population1 : this.population) {
            population1.fitness = 0;
        }

        double score = 0;
        //lowest score
        Network net;
        for (double ex = 0; ex <= 6; ex += 2) { // loops exponent
            for (int testSize = 2; testSize < 100; testSize += 5) {
                net = new Network(this.ActivationFunction, 4, 4, 1);
                double[] inData = new double[net.INPUT_SIZE];
                double[] outData = new double[net.OUTPUT_SIZE];
                for (int data = 0; data < testSize; data++) { // add some data
                    for (int i = 0; i < inData.length; i++) {
                        inData[i] = Math.random();
                    }
                    for (int i = 0; i < outData.length; i++) {
                        outData[i] = Math.random();
                    }
                }
                for (int i = 0; i < Math.pow(10, ex); i++) {
                    net.train(inData, outData, 0.3);
                }
                for (GeneticNetwork population1 : this.population) {
                    Network testNet = Network.copy(net);
                    testNet.train(inData, outData, population1.calculate(new double[]{testNet.NETWORK_SIZE, testSize, net.MSE(inData, outData)})[0]);
                    //System.out.println(Arrays.toString(inData) + Arrays.toString(outData));
                    population1.fitness += testNet.MSE(inData, outData);
                }
            }
        }
    }

    public void FitnessBase() {
        double score = 0;
        //lowest score
        Network net;
        for (double ex = 0; ex <= 5; ex += 1) {
            for (int testSize = 1; testSize < 76; testSize += 5) {
                double[] inData = new double[4];
                double[] outData = new double[1];
                for (int data = 0; data < testSize; data++) { // add some data
                    for (int i = 0; i < inData.length; i++) {
                        inData[i] = (i + 1) * 100;
                    }
                    for (int i = 0; i < outData.length; i++) {
                        outData[i] = 1d / (i + 1);
                    }
                }
                double mse;
                do {
                    net = new Network(Network.ZERO_TO_ONE, new int[]{4, 4, 1});
                    mse = net.MSE(inData, outData);
                    for (int i = 0; i < Math.pow(10, ex); i++) {
                        net.train(inData, outData, Network.LEARNING_RATE);
                    }
                } while (Double.isNaN(net.MSE(inData, outData)));
                //score +=  ?  ?  ?;
            }
        }
        System.out.println(score);
    }

    public GeneticNetwork[] highestFitnessNetworks(int topX, boolean isHighestScoreBest) {
    	GeneticNetwork[] top = new GeneticNetwork[topX];
        if (isHighestScoreBest) {
            for (int i = 0; i < top.length; i++) {
                top[i] = new GeneticNetwork(this.ActivationFunction, this.population[0].NETWORK_LAYER_SIZES);
                top[i].fitness = 0.0;
            }
            for (GeneticNetwork thePopulation : this.population) {
                int score;
                for (score = 0; score < top.length; score++) {
                    // top score is 0, lowest is top.length
                    if (thePopulation.fitness > top[score].fitness) {
                        break;
                    }
                }
                if (score < top.length) {
                    for (int j = top.length - 1; j > score; j--) { //move networks down
                        top[j] = top[j - 1];
                    }
                    top[score] = thePopulation; // put score into array
                }
            }
        } else {
            for (int i = 0; i < top.length; i++) {
                top[i] = new GeneticNetwork(this.ActivationFunction, this.population[0].NETWORK_LAYER_SIZES);
                top[i].fitness = Double.MAX_VALUE;
            }
            for (GeneticNetwork thePopulation : this.population) {
                int score;
                for (score = 0; score < top.length; score++) {
                    if (thePopulation.fitness < top[score].fitness) {
                        break;
                    }
                }
                if (score < top.length) {
                    for (int j = top.length - 1; j > score; j--) { //move networks down
                        top[j] = top[j - 1];
                    }
                    top[score] = thePopulation; // put score into array
                }
            }
        }
        for (int i = 0; i < top.length; i++) {
            try {
                top[i].saveNetwork(new File("").getAbsolutePath() + "\\src\\saves\\top" + i + "-2.txt");
                System.out.println(top[i].fitness);
            } catch (Exception ex) {
                System.err.println("failed to save: " + ex);
            }
        }
        return top;
    }

    public void nextGeneration() {
    	GeneticNetwork[] bestNetworks = this.highestFitnessNetworks(5, false);
        GeneticNetwork[] nextGeneration = new GeneticNetwork[this.population.length];
        System.arraycopy(bestNetworks, 0, nextGeneration, 0, bestNetworks.length);

        for (int i = bestNetworks.length; i < nextGeneration.length - 5; i++) { // fill next generation with children
            for (GeneticNetwork bestNetwork : bestNetworks) {
                nextGeneration[i] = this.breed(bestNetworks[i % bestNetworks.length], bestNetwork);
            }
        }
        for (int i = nextGeneration.length - 5; i < nextGeneration.length; i++) { // adds some diversity every time
            nextGeneration[i] = new GeneticNetwork(this.ActivationFunction, nextGeneration[i - 1].NETWORK_LAYER_SIZES);
        }
        this.population = nextGeneration;
    }

    public Network breed(GeneticNetwork parentA, GeneticNetwork parentB) {
        if (!Arrays.equals(parentA.NETWORK_LAYER_SIZES, parentB.NETWORK_LAYER_SIZES)) {
            return null;
        }
        DNA dna_a = new DNA(parentA);
        DNA dna_b = new DNA(parentB);

        GeneticNetwork child = new GeneticNetwork(this.ActivationFunction, parentA.NETWORK_LAYER_SIZES);

        for (int i = 0; i < child.bias.length; i++) {
            for (int j = 1; j < child.bias[i].length; j++) {
                //breed the parent a and b
                String a = dna_a.biasCromosomes[i].genes[j]; //get binary strings
                String b = dna_b.biasCromosomes[i].genes[j];
                String childGene = "";

                int index = 0;
                boolean isOnA = true;
                int larger;
                if (a.length() > b.length()) {
                    larger = a.length();
                } else {
                    larger = b.length();
                }
                while (index < larger - 1) {
                    int random;
                    //make sure random is not longer than possible
                    do {
                        random = (int) (Math.random() * (larger - index) + 1);
                    } while (random == 0);
                    if (isOnA) { // alternate parents
                        isOnA = false;
                        if (random + index < a.length()) {
                            childGene += a.substring(index, index + random + 1);
                            index += random;
                        }
                    } else {
                        isOnA = true;
                        if (random + index < b.length()) {
                            childGene += b.substring(index, index + random + 1);
                            index += random;
                        }
                    }
                }
                child.bias[i][j] = this.binStrToDbl(this.mutate(childGene, Population.MUTATION_RATE));
            }
        }

        for (int i = 1; i < child.weights.length; i++) {
            for (int j = 0; j < child.weights[i].length; j++) {
                for (int l = 0; l < child.weights[i][j].length; l++) {

                    //breed the parent a and b
                    String a = dna_a.biasCromosomes[i].genes[j]; //get binary strings
                    String b = dna_b.biasCromosomes[i].genes[j];
                    String childGene = "";

                    int index = 0;
                    boolean isOnA = true;
                    int larger;
                    if (a.length() > b.length()) {
                        larger = a.length();
                    } else {
                        larger = b.length();
                    }
                    while (index < larger - 1) {
                        int random;
                        //make sure random is not longer than possible
                        do {
                            random = (int) (Math.random() * (larger - index) + 1);
                        } while (random == 0);
                        if (isOnA) { // alternate parents
                            isOnA = false;
                            if (random + index < a.length()) {
                                childGene += a.substring(index, index + random + 1);
                                index += random;
                            }
                        } else {
                            isOnA = true;
                            if (random + index < b.length()) {
                                childGene += b.substring(index, index + random + 1);
                                index += random;
                            }
                        }
                    }
                    child.weights[i][j][l] = this.binStrToDbl(this.mutate(childGene, Population.MUTATION_RATE));
                }
            }
        }

        return child;
    }

    public String mutate(String gene, double chance) {
        String mutated = "";
        for (int i = 0; i < gene.length() - 1; i++) {
            if (Math.random() < chance) {
                if (gene.substring(i, i + 1).equals("1")) {
                    mutated += "0";
                } else {
                    mutated += "1";
                }
            } else {
                mutated += gene.substring(i, i + 1);
            }
        }
        return mutated;
    }

    private double binStrToDbl(String myBinStr) {
        return new BigInteger(myBinStr, 2).doubleValue();
    }
}
