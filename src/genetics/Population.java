package genetics;

import java.io.File;
import java.util.Arrays;

import fullyConnectedNetwork.ActivationFunction;
import fullyConnectedNetwork.GroupCalculate;
import fullyConnectedNetwork.ActivationFunction.Sigmoid;
import fullyConnectedNetwork.Network;
import fullyConnectedNetwork.NetworkGroup;

abstract public class Population {

    public NetworkGroup[] population;
    private static final int TOP_NETWORKS_NUM = 5;
    private static final int DIVERSITY_RATING = 1; // how many new networks get added each generation
    private static final double MUTATION_RATE = 1.0; // max percent change from original value.
    private static final boolean isHighestScoreBest = false;
    private static final String path = "C:\\Users\\redfl\\Documents\\NetworkSaves\\GeneticSaves\\";

    // we clone the network layer sizes in each constructor so that we get different
    // instances of the array instead of multiple instances of the same array
    public Population(int popSize, ActivationFunction af, double mutiplier, int... NETWORK_LAYER_SIZES) {
        this.population = new NetworkGroup[popSize];
        for (int i = 0; i < this.population.length; i++) {
            this.population[i] = new NetworkGroup(
                    new GeneticNetwork[][] { { new GeneticNetwork(af, mutiplier, NETWORK_LAYER_SIZES.clone()) } },
                    new SingleNetworkClaculate()); // automatically randomly generates new and unique networks
        }
    }

    public Population(int popSize, ActivationFunction af, int adjustingNeurons, double mutiplier,
            int... NETWORK_LAYER_SIZES) {
        this.population = new NetworkGroup[popSize];
        for (int i = 0; i < this.population.length; i++) {
            this.population[i] = new NetworkGroup(
                    new SelfAdjustingNetwork[][] { {
                            new SelfAdjustingNetwork(af, adjustingNeurons, mutiplier, NETWORK_LAYER_SIZES.clone()) } },
                    new SingleNetworkClaculate()); // automatically randomly generates new and unique networks
        }
    }

    public Population(int popSize, ActivationFunction af, int memoryLength, int memoryWidth, double mutiplier,
            int... NETWORK_LAYER_SIZES) {
        this.population = new NetworkGroup[popSize];
        for (int i = 0; i < this.population.length; i++) {
            this.population[i] = new NetworkGroup(new GeneticMemoryNetwork[][] { {
                    new GeneticMemoryNetwork(af, memoryLength, memoryWidth, mutiplier, NETWORK_LAYER_SIZES.clone()) } },
                    new SingleNetworkClaculate()); // automatically randomly generates new and unique networks
        }
    }

    public Population(int popSize, ActivationFunction af, int adjustingNeurons, int memoryLength, int memoryWidth,
            double mutiplier, int... NETWORK_LAYER_SIZES) {
        this.population = new NetworkGroup[popSize];
        for (int i = 0; i < this.population.length; i++) {
            this.population[i] = new NetworkGroup(new Brain[][] { { new Brain(af, adjustingNeurons, memoryLength,
                    memoryWidth, mutiplier, NETWORK_LAYER_SIZES.clone()) } }, new SingleNetworkClaculate()); // automatically
                                                                                                             // randomly
                                                                                                             // generates
                                                                                                             // new and
                                                                                                             // unique
                                                                                                             // networks
        }
    }

    public Population(int popSize, Network[][] group, GroupCalculate GC) {
        this.population = new NetworkGroup[popSize];

        for (int i = 0; i < this.population.length; i++) {
            this.population[i] = new NetworkGroup(group, GC); // automatically randomly generates new and unique
                                                              // networks
        }
    }

    private class SingleNetworkClaculate extends GroupCalculate {

        @Override
        public double[] calculate(Network[][] networkGroups, double... input) {
            return networkGroups[0][0].calculate(input);
        }

    }

    public static void main(String[] args) {
        /*
         * class dualNetwork extends GroupCalculate {
         * 
         * @Override public double[] calculate(Network[][] group, double... input) {
         * double[] output = group[0][0].calculate(input); return
         * group[1][0].calculate(output); // use the output of the previous network as
         * the input for the second one }
         * 
         * }
         */

        class population extends Population {
            public population(int popSize, Network[][] group, GroupCalculate GC) {
                super(popSize, group, GC);
            }

            public population(int popSize, ActivationFunction af, int adjustingNeurons, double mutiplier,
                    int... NETWORK_LAYER_SIZES) {
                super(popSize, af, adjustingNeurons, mutiplier, NETWORK_LAYER_SIZES);
            }

            public population(int popSize, ActivationFunction af, int memoryLength, int memoryWidth, double multiplier,
                    int... NETWORK_LAYER_SIZES) {
                super(popSize, af, memoryLength, memoryWidth, multiplier, NETWORK_LAYER_SIZES);
            }

            @Override
            public void Fitness() {
                for (NetworkGroup pop : this.population) {
                    ((GeneticMemoryNetwork) pop.group[0][0]).clearMemory();
                    pop.groupFitness = 0.0;
                    for (int i = 0; i < 3; i++) {
                        double target = ((double) i ) / 3;
                        pop.groupFitness += pop.group[0][0].MSE(new double[]{2}, new double[]{target});
                    }
                }
            }
        }

        population pop = new population(75, new ActivationFunction.Sigmoid(), 5, 2, 1.0, 1, 1, 1);
        /*
         * Network[][] group = new Network[][] { {new SelfAdjustingNetwork(new
         * ActivationFunction.Sigmoid(), 2, 3, 3, 3)}, {new GeneticMemoryNetwork(new
         * ActivationFunction.Sigmoid(), 10, 1, 10.0, 3, 2, 2)} }; population pop = new
         * population(72, group, new dualNetwork()); // initialize population
         * 
         * for (int i = 0; i < 5; i++) { // load current top 5 networks try {
         * System.out.println(Population.path + i + "\\"); pop.population[i] = new
         * NetworkGroup(Population.path + i + "\\", new dualNetwork()); } catch
         * (Exception ex) { System.err.println(Arrays.toString(ex.getStackTrace())); } }
         */
        int i = 0;
        while (i < 1000) { // repeat
            pop.Fitness(); // evaluate fitness
            System.out.println("Fitness complete");
            pop.nextGeneration(); // replace old generation with new one
            System.out.println("generation killed");
            i++;
        }
        ((GeneticMemoryNetwork) pop.population[0].group[0][0]).clearMemory();
        for(i = 0; i < 3; i++){
            System.out.println(Arrays.toString(pop.population[0].calculate(2)));
        }
    }

    abstract public void Fitness();

    public NetworkGroup[] highestFitnessNetworks(int topX) {
        NetworkGroup[] top = new NetworkGroup[topX];

        class filler extends GroupCalculate {
            @Override
            public double[] calculate(Network[][] networkGroups, double... input) {
                return null;
            }

        }

        if (Population.isHighestScoreBest) {
            for (int i = 0; i < top.length; i++) {
                top[i] = new NetworkGroup(new Network[][] { {} }, new filler());
                top[i].groupFitness = Double.NEGATIVE_INFINITY;
            }
            for (NetworkGroup thePopulation : this.population) {
                int score;
                for (score = 0; score < top.length; score++) {
                    // top score is 0, lowest is top.length
                    if (thePopulation.groupFitness > top[score].groupFitness) {
                        break;
                    }
                }
                if (score < top.length) {
                    for (int j = top.length - 1; j > score; j--) { // move networks down
                        top[j] = top[j - 1];
                    }
                    top[score] = thePopulation; // put score into array
                }
            }
        } else {
            for (int i = 0; i < top.length; i++) {
                top[i] = new NetworkGroup(new Network[][] { {} }, new filler());
                top[i].groupFitness = Double.POSITIVE_INFINITY;
            }
            for (NetworkGroup thePopulation : this.population) {
                int score;
                for (score = 0; score < top.length; score++) {
                    if (thePopulation.groupFitness < top[score].groupFitness) {
                        break;
                    }
                }
                if (score < top.length) {
                    for (int j = top.length - 1; j > score; j--) { // move networks down
                        top[j] = top[j - 1];
                    }
                    top[score] = thePopulation; // put score into array
                }
            }
        }
        for (int i = 0; i < top.length; i++) {
            new File(path + i).mkdirs();
            top[i].saveNetworkGroup(path + i);
            System.out.println(top[i].groupFitness);
        }
        return top;
    }

    public void nextGeneration() {
        NetworkGroup[] bestNetworks = this.highestFitnessNetworks(Population.TOP_NETWORKS_NUM);
        NetworkGroup[] parents = new NetworkGroup[bestNetworks.length + Population.DIVERSITY_RATING];

        NetworkGroup[] nextGeneration = new NetworkGroup[this.population.length];
        System.arraycopy(bestNetworks, 0, nextGeneration, 0, bestNetworks.length);
        System.arraycopy(bestNetworks, 0, parents, 0, bestNetworks.length);

        for (int i = bestNetworks.length; i < parents.length; i++) {
            parents[i] = NetworkGroup.cloneAndRandomize(bestNetworks[0]); // which network doesn't matter as all will
                                                                          // share the same Calculate() method
        }

        int parentLengthMinOne = parents.length - 1;
        int i = parentLengthMinOne - 1;
        int next = 0;
        while (i < nextGeneration.length - 1) {
            nextGeneration[++i] = this.breed(parents[i % parentLengthMinOne], parents[next % parentLengthMinOne]);
            if (i % (parentLengthMinOne) == 0) {
                next++;
            }
        }

        this.population = nextGeneration;
    }

    public NetworkGroup breed(NetworkGroup parentA, NetworkGroup parentB) {
        if (!parentA.equals(parentB)) {
            return null;
        }
        NetworkGroup child = NetworkGroup.cloneAndRandomize(parentA);
        for (int netArr = 0; netArr < child.group.length; netArr++) {
            for (int childNet = 0; childNet < child.group[netArr].length; childNet++) {

                double percentWeight = (parentA.groupFitness * 0.5) / parentB.groupFitness; // weigh the randomizer
                                                                                            // towards the better parent

                for (int i = 0; i < child.group[netArr][childNet].bias.length; i++) {
                    for (int j = 0; j < child.group[netArr][childNet].bias[i].length; j++) {
                        double parentVal = 0;
                        if (Math.random() < percentWeight) {
                            parentVal = parentA.group[netArr][childNet].bias[i][j];
                        } else {
                            parentVal = parentB.group[netArr][childNet].bias[i][j];
                        }
                        parentVal += (Math.random() * 2.0 - 1) * Population.MUTATION_RATE;
                        child.group[netArr][childNet].bias[i][j] = parentVal;
                        if (i > 0) {
                            for (int l = 0; l < child.group[netArr][childNet].weights[i][j].length; l++) {
                                parentVal = 0;
                                if (Math.random() < percentWeight) {
                                    parentVal = parentA.group[netArr][childNet].weights[i][j][l];
                                } else {
                                    parentVal = parentB.group[netArr][childNet].weights[i][j][l];
                                }
                                parentVal += (Math.random() * 2.0 - 1) * Population.MUTATION_RATE;
                                child.group[netArr][childNet].weights[i][j][l] = parentVal;
                            }
                        }
                    }
                }

                if (child.group[netArr][childNet] instanceof SelfAdjustingNetwork) {
                    SelfAdjustingNetwork.breed((SelfAdjustingNetwork) parentA.group[netArr][childNet],
                            (SelfAdjustingNetwork) parentB.group[netArr][childNet],
                            (SelfAdjustingNetwork) child.group[netArr][childNet], Population.MUTATION_RATE);
                } else if (child.group[netArr][childNet] instanceof Brain) {
                    Brain.breed((Brain) parentA.group[netArr][childNet], (Brain) parentB.group[netArr][childNet],
                            (Brain) child.group[netArr][childNet], Population.MUTATION_RATE);
                }

            }
        }
        return child;
    }
}
