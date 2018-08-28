package genetics;

import java.util.Arrays;

import fullyConnectedNetwork.Network;

abstract public class Population {

    public GeneticNetwork[] population;
    //public final double[] target = new double[]{1, 1, 0, 0};
    private final int memWidth, memLength, adjustingNeurons;
    private final int[] NETWORK_LAYER_SIZES;
    private static final int TOP_NETWORKS_NUM = 5;
    private static final int DIVERSITY_RATING = 1; // how many new networks get added each generation
    private static final double MUTATION_RATE = 0.1; //max percent change from original value.
    private static final boolean isHighestScoreBest = false;
    private static final String path = "C:\\Users\\Home-Lucas\\Documents\\Saves\\NeuralNetworks\\GeneticNetworks\\";
    private final int ActivationFunction;
    private final Class<?> c;

    //we clone the network layer sizes in each constructor so that we get different instances of the array instead of multiple instances of the same array
    public Population(int popSize, int af, double mutiplier, int... NETWORK_LAYER_SIZES) {
    	this.c = GeneticNetwork.class;
    	this.adjustingNeurons = this.memLength = this.memWidth = 0;
        this.population = new GeneticNetwork[popSize];
        this.ActivationFunction = af;
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        for (int i = 0; i < this.population.length; i++) {
            this.population[i] = new GeneticNetwork(this.ActivationFunction, mutiplier, this.NETWORK_LAYER_SIZES.clone()); // automatically randomly generates new and unique networks
        }
    }
    
    public Population(int popSize, int af, int adjustingNeurons, double mutiplier, int... NETWORK_LAYER_SIZES) {
    	this.c = SelfAdjustingNetwork.class;
    	this.memLength = this.memWidth = 0;
    	this.adjustingNeurons = adjustingNeurons;
        this.population = new SelfAdjustingNetwork[popSize];
        this.ActivationFunction = af;
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        for (int i = 0; i < this.population.length; i++) {
            this.population[i] = new SelfAdjustingNetwork(this.ActivationFunction, this.adjustingNeurons, mutiplier, this.NETWORK_LAYER_SIZES.clone()); // automatically randomly generates new and unique networks
        }
    }
    
    public Population(int popSize, int af, int memoryLength, int memoryWidth, double mutiplier, int... NETWORK_LAYER_SIZES) {
    	this.c = GeneticMemoryNetwork.class;
    	this.memWidth = memoryWidth;
    	this.memLength = memoryLength;
    	this.adjustingNeurons = 0;
        this.population = new GeneticMemoryNetwork[popSize];
        this.ActivationFunction = af;
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        for (int i = 0; i < this.population.length; i++) {
            this.population[i] = new GeneticMemoryNetwork(this.ActivationFunction, this.memLength, this.memWidth, mutiplier, this.NETWORK_LAYER_SIZES.clone()); // automatically randomly generates new and unique networks
        }
    }

    public Population(int popSize, int af, int adjustingNeurons, int memoryLength, int memoryWidth, double mutiplier, int... NETWORK_LAYER_SIZES) {
    	this.c = Brain.class;
    	this.memWidth = memoryWidth;
    	this.memLength = memoryLength;
    	this.adjustingNeurons = adjustingNeurons;
        this.population = new Brain[popSize];
        this.ActivationFunction = af;
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        for (int i = 0; i < this.population.length; i++) {
            this.population[i] = new Brain(this.ActivationFunction, this.adjustingNeurons, this.memLength, this.memWidth, mutiplier, this.NETWORK_LAYER_SIZES.clone()); // automatically randomly generates new and unique networks
        }
    }
    
    public static void main(String[] args) {
        class population extends Population{

			public population(int popSize, int af, int adjustingNeurons, int memoryLength, int memoryWidth, double mutiplier, int... NETWORK_LAYER_SIZES) {
				super(popSize, af, adjustingNeurons, memoryLength, memoryWidth, mutiplier, NETWORK_LAYER_SIZES);
				// TODO Auto-generated constructor stub
			}

			@Override
			public void Fitness() {
		    	for(GeneticNetwork pop: this.population) {
		    		pop.fitness = 0;
		    		for(int i = 0; i < 10; i++) {
		    			pop.fitness += Math.abs(((3 * i) / 2) - pop.calculate(new double[] {i, i * 2})[0]);
		    		}
		    	}
			}
        }
        
        population pop = new population(72, Network.ZERO_TO_ONE, 2, 10, 2, 10.0, 2, 2, 1); // initialize population
        /*
        for (int i = 0; i < 5; i++) { // load current top 5 networks
        	try {
        		pop.population[i] = GeneticMemoryNetwork.loadNetwork(path + i + ".txt");
        	} catch (Exception ex) {
        		System.err.println(Arrays.toString(ex.getStackTrace()));
        	}
        }*/
        
        int i = 0;
        while (i < 1000) { // repeat 
            pop.Fitness(); // evaluate fitness
            //pop.FitnessBase(); // compare to base
            System.out.println("Fitness complete");
            pop.nextGeneration(); // replace old generation with new one
            System.out.println("generation killed");
            i++;
        }
        
    }

    abstract public void Fitness();

    public GeneticNetwork[] highestFitnessNetworks(int topX) {
    	GeneticNetwork[] top = new GeneticNetwork[topX];
        if (Population.isHighestScoreBest) {
            for (int i = 0; i < top.length; i++) {
                top[i] = new GeneticNetwork(this.ActivationFunction, this.NETWORK_LAYER_SIZES.clone());
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
                top[i] = new GeneticNetwork(this.ActivationFunction, this.NETWORK_LAYER_SIZES.clone());
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
        	top[i].saveNetwork(path + i + ".txt");
            System.out.println(top[i].fitness);
        }
        return top;
    }

    public void nextGeneration() {
    	GeneticNetwork[] bestNetworks = this.highestFitnessNetworks(Population.TOP_NETWORKS_NUM);
    	GeneticNetwork[] parents = new GeneticNetwork[bestNetworks.length + Population.DIVERSITY_RATING];
    	
        GeneticNetwork[] nextGeneration = new GeneticNetwork[this.population.length];
        System.arraycopy(bestNetworks, 0, nextGeneration, 0, bestNetworks.length);
        System.arraycopy(bestNetworks, 0, parents, 0, bestNetworks.length);
    	
    	for(int i = bestNetworks.length; i < parents.length; i++) {
        	if(this.c.equals(GeneticNetwork.class)) {
        		parents[i] = new GeneticNetwork(this.ActivationFunction, this.population[0].multiplier, this.NETWORK_LAYER_SIZES.clone()); 
        	}else if(this.c.equals(GeneticMemoryNetwork.class)) {
        		parents[i] = new GeneticMemoryNetwork(this.ActivationFunction, this.memLength, this.memWidth, this.population[0].multiplier, this.NETWORK_LAYER_SIZES.clone());
        	}else if(this.c.equals(SelfAdjustingNetwork.class)) {
        		parents[i] = new SelfAdjustingNetwork(this.ActivationFunction, this.adjustingNeurons, this.population[0].multiplier, this.NETWORK_LAYER_SIZES.clone()); 
        	}else if(this.c.equals(Brain.class)) {
        		parents[i] = new Brain(this.ActivationFunction, this.adjustingNeurons, this.memLength, this.memWidth, this.population[0].multiplier, this.NETWORK_LAYER_SIZES.clone()); 
        	}
    	}
        
    	int bestLengthMinOne = bestNetworks.length - 1;
        int i = bestLengthMinOne;
        int next = -1;
        while(i < nextGeneration.length - 1){
        	if(i % (bestLengthMinOne) == 0) {
        		next++;
        	}
        	nextGeneration[++i] = this.breed(bestNetworks[i % bestLengthMinOne], bestNetworks[next % bestLengthMinOne]);
        }
        
        this.population = nextGeneration;
    }

    public GeneticNetwork breed(GeneticNetwork parentA, GeneticNetwork parentB) {
        if (!Arrays.equals(parentA.NETWORK_LAYER_SIZES, parentB.NETWORK_LAYER_SIZES)) {
            return null;
        }
        GeneticNetwork child;
    	if(this.c.equals(GeneticNetwork.class)) {
    		child = new GeneticNetwork(this.ActivationFunction, this.population[0].multiplier, this.NETWORK_LAYER_SIZES.clone()); 
    	}else if(this.c.equals(GeneticMemoryNetwork.class)) {
    		child = new GeneticMemoryNetwork(this.ActivationFunction, this.memLength, this.memWidth, this.population[0].multiplier, this.NETWORK_LAYER_SIZES.clone());
    	}else if(this.c.equals(SelfAdjustingNetwork.class)) {
    		child = new SelfAdjustingNetwork(this.ActivationFunction, this.adjustingNeurons, this.population[0].multiplier, this.NETWORK_LAYER_SIZES.clone()); 
    	}else if(this.c.equals(Brain.class)) {
    		child = new Brain(this.ActivationFunction, this.adjustingNeurons, this.memLength, this.memWidth, this.population[0].multiplier, this.NETWORK_LAYER_SIZES.clone()); 
    	}else {
    		child = null;
    	}

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
        
        if(this.c.equals(SelfAdjustingNetwork.class)) {
        	SelfAdjustingNetwork.breed( (SelfAdjustingNetwork) parentA, (SelfAdjustingNetwork) parentB, (SelfAdjustingNetwork) child,  Population.MUTATION_RATE);
        }else if(this.c.equals(Brain.class)) {
        	Brain.breed( (Brain) parentA, (Brain) parentB, (Brain) child,  Population.MUTATION_RATE);
        }

        return child;
    }

}
