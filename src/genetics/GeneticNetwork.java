package genetics;

import fullyconnectednetwork.Network;

public class GeneticNetwork extends Network {

	double fitness;
	
	public GeneticNetwork(int ActivationFunction, double multiplier, int[] NETWORK_LAYER_SIZES) {
		super(ActivationFunction, multiplier, NETWORK_LAYER_SIZES);
		this.fitness = 0;
	}
	
	public GeneticNetwork(int ActivationFunction, int[] NETWORK_LAYER_SIZES) {
		super(ActivationFunction, NETWORK_LAYER_SIZES);
		this.fitness = 0;
	}

}
