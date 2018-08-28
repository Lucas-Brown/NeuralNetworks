package fullyConnectedNetwork;

public abstract class NetworkGroup {
	
	public Network[][] group;
	public double[] groupOutput;
	
	public NetworkGroup(Network[][] networks) {
		this.group = networks;
	}
	
	abstract public double[] calculate(double... input);
}
