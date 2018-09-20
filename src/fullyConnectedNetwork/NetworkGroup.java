package fullyConnectedNetwork;

import java.io.File;
import java.io.FileFilter;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Scanner;

import genetics.Brain;
import genetics.GeneticMemoryNetwork;
import genetics.GeneticNetwork;
import genetics.SelfAdjustingNetwork;

public class NetworkGroup{
	
	public Network[][] group;
	private Class<?>[][] c; 
	public final int[] NETWORK_LAYER_SIZES;
	public Double groupFitness;
	private GroupCalculate GC;
	
	public NetworkGroup(Network[][] networks, GroupCalculate GC) {
		this.GC = GC;
		this.group = networks;
		this.NETWORK_LAYER_SIZES = new int[this.group.length];
		for(int i = 0; i < this.NETWORK_LAYER_SIZES.length; i++) {
			this.NETWORK_LAYER_SIZES[i] = this.group[i].length;
		}
		this.c = new Class[this.NETWORK_LAYER_SIZES.length][];
		for(int i = 0; i < this.NETWORK_LAYER_SIZES.length; i++) {
			this.c[i] = new Class[this.NETWORK_LAYER_SIZES[i]];
			for(int net = 0; net < this.NETWORK_LAYER_SIZES[i]; net++) {
				Network netCheck = this.group[i][net];
				
				if(netCheck instanceof Brain) {
					this.c[i][net] = Brain.class;
				}else if(netCheck instanceof SelfAdjustingNetwork) {
					this.c[i][net] = SelfAdjustingNetwork.class;
				}else if(netCheck instanceof GeneticMemoryNetwork) {
					this.c[i][net] = GeneticMemoryNetwork.class;
				}else if(netCheck instanceof GeneticNetwork) {
					this.c[i][net] = GeneticNetwork.class;
				}else if(netCheck instanceof Network) {
					this.c[i][net] = Network.class;
				}
			}
		}
	}
	
	public NetworkGroup(String filePath, GroupCalculate GC) {
		this.GC = GC;
		Network[][] group;
		
		File folder = new File(filePath);
		File[] folders = folder.listFiles(new FileFilter() {
		    @Override
		    public boolean accept(File f) {
		        return f.isDirectory(); // make sure its a folder
		    }
		});
		group = new Network[folders.length][];
		
		this.c = new Class[group.length][];
		for(int layer = 0; layer < group.length; layer++) {
			String layerDir = filePath + "\\layer-" + (layer + 1);
			
			File file = new File(layerDir);
			File[] files = file.listFiles(new FileFilter() {
			    @Override
			    public boolean accept(File f) {
			        return f.isFile(); // make sure its a file
			    }
			});
			group[layer] = new Network[files.length];
			
			this.c[layer] = new Class[group[layer].length];
			for(int network = 0; network < group[layer].length; network++) {
				String networkFilePath = layerDir + "\\network-" + (network + 1);
				Class<?> c = NetworkGroup.networkType(networkFilePath);
				this.c[layer][network] = c;
				try {
					if(c.equals(Network.class)) {
						group[layer][network] = Network.loadNetwork(networkFilePath);
					} else if(c.equals(GeneticNetwork.class)) {
						group[layer][network] = GeneticNetwork.loadNetwork(networkFilePath);
					} else if(c.equals(GeneticMemoryNetwork.class)) {
						group[layer][network] = GeneticMemoryNetwork.loadNetwork(networkFilePath);
					} else if(c.equals(SelfAdjustingNetwork.class)) {
						group[layer][network] = SelfAdjustingNetwork.loadNetwork(networkFilePath);
					} else if(c.equals(Brain.class)) {
						group[layer][network] = Brain.loadNetwork(networkFilePath);
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}
		this.group = group;
		this.NETWORK_LAYER_SIZES = new int[this.group.length];
		for(int i = 0; i < this.NETWORK_LAYER_SIZES.length; i++) {
			this.NETWORK_LAYER_SIZES[i] = this.group[i].length;
		}
	}
	
	public void saveNetworkGroup(String filePath) {
		//String fullName = this.toString();
		//String name = fullName.substring(fullName.indexOf("$") + 2, fullName.indexOf("@"));
		String groupDir = filePath; //+ "\\" + fullName;
		new File(groupDir).mkdirs();
		
		for(int layer = 0; layer < this.group.length; layer++) {
			String layerDir = groupDir + "\\layer-" + (layer + 1);
			new File(layerDir).mkdirs();
			for(int network = 0; network < this.group[layer].length; network++) {
				this.group[layer][network].saveNetwork(layerDir + "\\network-" + (network + 1));
			}
		}	
	}
	
	private static Class<?> networkType(String filePath){
		Class<?>[] c = new Class<?>[] {GeneticNetwork.class, GeneticMemoryNetwork.class, SelfAdjustingNetwork.class, Brain.class}; 
		try {
			for(int i = 0; i < c.length; i++) {
				Scanner sc = new Scanner(new File(filePath));
				String fullClassName = c[i].toString();
				String className = fullClassName.substring(fullClassName.indexOf(".") + 1);
				if (className == sc.findInLine(className)) {
					sc.close();
					return c[i];
				}else {
					sc.close();
				}
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return Network.class;
	}
	
	public boolean equals(NetworkGroup ng) {
		if(!Arrays.equals(this.NETWORK_LAYER_SIZES, ng.NETWORK_LAYER_SIZES)) {
			return false;
		}
		for(int i = 0; i < this.NETWORK_LAYER_SIZES.length; i++) {
			for(int j = 0; j < this.NETWORK_LAYER_SIZES[i]; j++) {
				if(!Arrays.equals(this.group[i][j].NETWORK_LAYER_SIZES, ng.group[i][j].NETWORK_LAYER_SIZES)) {
					return false;
				}
			}
		}
		return true;
	}
	
	public static NetworkGroup cloneAndRandomize(NetworkGroup netGroup) {
		Network[][] newGroup = new Network[netGroup.group.length][];
		
		for(int i = 0; i < newGroup.length; i++) {
			newGroup[i] = new Network[netGroup.NETWORK_LAYER_SIZES[i]];
			for(int net = 0; net < newGroup[i].length; net++) {
				
				if(netGroup.c[i][net].equals(Network.class)) {
					Network selectedNetwork = netGroup.group[i][net];
					newGroup[i][net] = new Network(selectedNetwork.ACTIVATION_FUNCTION, selectedNetwork.multiplier, selectedNetwork.NETWORK_LAYER_SIZES.clone());
				}else if(netGroup.c[i][net].equals(GeneticNetwork.class)) {
					GeneticNetwork network = (GeneticNetwork) netGroup.group[i][net];
					newGroup[i][net] = new GeneticNetwork(network.ACTIVATION_FUNCTION, network.multiplier, network.NETWORK_LAYER_SIZES.clone());
				}else if(netGroup.c[i][net].equals(SelfAdjustingNetwork.class)) {
					SelfAdjustingNetwork network = (SelfAdjustingNetwork) netGroup.group[i][net];
					newGroup[i][net] = new SelfAdjustingNetwork(network.ACTIVATION_FUNCTION, network.adjustingNeurons, network.multiplier, 
							SelfAdjustingNetwork.reverseAdjustLayers(network.adjustingNeurons, network.NETWORK_LAYER_SIZES.clone()));
				}else if(netGroup.c[i][net].equals(GeneticMemoryNetwork.class)) {
					GeneticMemoryNetwork network = (GeneticMemoryNetwork) netGroup.group[i][net];
					newGroup[i][net] = new GeneticMemoryNetwork(network.ACTIVATION_FUNCTION, network.memoryLength, network.memoryWidth, network.multiplier, 
							GeneticMemoryNetwork.reverseAdjustLayers(network.memoryLength, network.memoryWidth, network.NETWORK_LAYER_SIZES.clone()));
				}else if(netGroup.c[i][net].equals(Brain.class)) {
					Brain network = (Brain) netGroup.group[i][net];
					newGroup[i][net] = new Brain(network.ACTIVATION_FUNCTION, network.adjustingNeurons, network.memoryLength, network.memoryWidth, network.multiplier, 
							Brain.reverseAdjustLayers(network.adjustingNeurons, network.memoryLength, network.memoryWidth, network.NETWORK_LAYER_SIZES.clone()));
				}
				
			}
		}
		return new NetworkGroup(newGroup, netGroup.GC.clone());
	}
	
	public double[] calculate(double... input) {
		return this.GC.calculate(this.group, input);
	}

	public double MSE(double[] input, double[] target) {
        double[] output = this.calculate(input);
        double v = 0;
        for (int i = 0; i < target.length; i++) {
            v += (target[i] - output[i]) * (target[i] - output[i]);
        }
        return v / (2d * target.length);
    }
}
