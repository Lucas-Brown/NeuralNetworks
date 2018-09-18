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

public abstract class NetworkGroup implements Cloneable{
	
	public Network[][] group;
	public final int[] NETWORK_LAYER_SIZES;
	public double groupFitness;
	protected double[] groupOutput;
	
	public NetworkGroup(Network[][] networks) {
		this.group = networks;
		this.NETWORK_LAYER_SIZES = new int[this.group.length];
		for(int i = 0; i < this.NETWORK_LAYER_SIZES.length; i++) {
			this.NETWORK_LAYER_SIZES[i] = this.group[i].length;
		}
	}
	
	public NetworkGroup(String filePath) {
		Network[][] group;
		
		File folder = new File(filePath);
		File[] folders = folder.listFiles(new FileFilter() {
		    @Override
		    public boolean accept(File f) {
		        return f.isDirectory(); // make sure its a folder
		    }
		});
		group = new Network[folders.length][];
		
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
			
			for(int network = 0; network < group[layer].length; network++) {
				String networkFilePath = layerDir + "\\network-" + (network + 1);
				Class<?> c = NetworkGroup.networkType(networkFilePath);
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
	
	public NetworkGroup cloneGroup() {
		try {
			return (NetworkGroup) this.clone();
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
			return null;
		}
	}
	
	public static NetworkGroup cloneAndRandomize(NetworkGroup netGroup) {
		NetworkGroup newGroup = netGroup.cloneGroup();
		for(Network[] layers: netGroup.group) {
			for(Network net: layers) {
				for (int i = 0; i < net.NETWORK_SIZE; i++) {
					net.bias[i] = NetworkTools.createRandomArray(net.NETWORK_LAYER_SIZES[i],  -0.5, 0.7);
					if (i > 0) {
						net.weights[i] = NetworkTools.createRandomArray(net.NETWORK_LAYER_SIZES[i], net.NETWORK_LAYER_SIZES[i - 1], -0.5, 0.7);
					}
				}
			}
		}
		return newGroup;
	}
	
	abstract public double[] calculate(double... input);
}
