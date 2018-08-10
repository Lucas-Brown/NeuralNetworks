package genetics;

import fullyconnectednetwork.Network;
import java.io.File;
import java.math.BigInteger;
import java.util.Arrays;

public class Population {

    public GeneticNetwork[] population;
    //public final double[] target = new double[]{1, 1, 0, 0};
    public static final double MUTATION_RATE = 0.001; //max percent change from original value
    public static final boolean isHighestScoreBest = false;
    public static final String path = "C:\\Users\\Lucas Brown\\Documents\\NetworkSaves\\GeneticSaves";
    public final int ActivationFunction;

    public Population(int popSize, int af, double mutiplier, int... NETWORK_LAYER_SIZES) {
        this.population = new GeneticNetwork[popSize];
        this.ActivationFunction = af;
        for (int i = 0; i < this.population.length; i++) {
            this.population[i] = new GeneticNetwork(af, mutiplier, NETWORK_LAYER_SIZES); // automatically randomly generates new and unique networks
        }
    }

    public static void main(String[] args) {
        Population pop = new Population(75,Network.ZERO_TO_ONE, 10.0, 2, 2, 1); // initialize population
        
        for (int i = 0; i < 5; i++) { // load current top 5 networks
        	try {
        		pop.population[i] = GeneticNetwork.loadNetwork(path + i + ".txt");
        	} catch (Exception ex) {
        		System.err.println(ex);
        	}
        }
        
        while (true) { // repeat 
            pop.Fitness(); // evaluate fitness
            //pop.FitnessBase(); // compare to base
            System.out.println("Fitness complete");
            pop.nextGeneration(); // replace old generation with new one
            System.out.println("generation killed");
        }
    }

    public void Fitness() {
    	for(GeneticNetwork pop: this.population) {
    		pop.fitness = 0;
    		for(int i = 0; i < 10; i++) {
    			pop.fitness += Math.abs(((3 * i) / 2) - pop.calculate(new double[] {i, i * 2})[0]);
    		}
    	}
    }
/*
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
*/
    public GeneticNetwork[] highestFitnessNetworks(int topX) {
    	GeneticNetwork[] top = new GeneticNetwork[topX];
        if (Population.isHighestScoreBest) {
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
                top[i].saveNetwork(path + i + ".txt");
                System.out.println(top[i].fitness);
            } catch (Exception ex) {
                System.err.println("failed to save: " + ex);
            }
        }
        return top;
    }

    public void nextGeneration() {
    	GeneticNetwork[] bestNetworks = this.highestFitnessNetworks(5);
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

    public GeneticNetwork breed(GeneticNetwork parentA, GeneticNetwork parentB) {
        if (!Arrays.equals(parentA.NETWORK_LAYER_SIZES, parentB.NETWORK_LAYER_SIZES)) {
            return null;
        }

        GeneticNetwork child = new GeneticNetwork(this.ActivationFunction, parentA.multiplier, parentA.NETWORK_LAYER_SIZES);

        double percentWeight = (parentA.fitness * 0.5) / parentB.fitness; // weigh the randomizer towards the better parent
        
        for (int i = 0; i < child.bias.length; i++) {
            for (int j = 1; j < child.bias[i].length; j++) {
            	double parentVal = 0;
            	if(Math.random() < percentWeight) {
            		parentVal = parentA.bias[i][j];
            	} else {
            		parentVal = parentB.bias[i][j];
        		}
            	parentVal += (Math.random() * 2.0 - 1) * Population.MUTATION_RATE;
            	child.bias[i][j] = parentVal;
            }
        }

        for (int i = 1; i < child.weights.length; i++) {
            for (int j = 0; j < child.weights[i].length; j++) {
                for (int l = 0; l < child.weights[i][j].length; l++) {
                	double parentVal = 0;
                	if(Math.random() < percentWeight) {
                		parentVal = parentA.weights[i][j][l];
                	} else {
                		parentVal = parentB.weights[i][j][l];
            		}
                	parentVal += (Math.random() * 2.0 - 1) * Population.MUTATION_RATE;
                	child.weights[i][j][l] = parentVal;
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
