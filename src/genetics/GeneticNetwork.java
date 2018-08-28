package genetics;

import java.util.Arrays;

import fullyConnectedNetwork.Network;
import parser.Attribute;
import parser.Node;
import parser.Parser;
import parser.ParserTools;

public class GeneticNetwork extends Network {

	double fitness;
	
	public GeneticNetwork(int ActivationFunction, double multiplier, int... NETWORK_LAYER_SIZES) {
		super(ActivationFunction, multiplier, NETWORK_LAYER_SIZES);
		this.fitness = 0;
	}
	
	public GeneticNetwork(int ActivationFunction, int... NETWORK_LAYER_SIZES) {
		super(ActivationFunction, NETWORK_LAYER_SIZES);
		this.fitness = 0;
	}
	
	@Override
	public void saveNetwork(String fileName) {
        Parser p = new Parser();
        p.create(fileName);
        Node root = p.getContent();
        Node netw = new Node("Network");
        Node ly = new Node("Layers");
        netw.addAttribute(new Attribute("Activation Function", Integer.toString(this.ACTIVATION_FUNCTION)));
        netw.addAttribute(new Attribute("Multiplier", Double.toString(this.multiplier)));
        netw.addAttribute(new Attribute("fitness", Double.toString(this.fitness)));
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

    public static GeneticNetwork loadNetwork(String fileName) throws Exception {

        Parser p = new Parser();

        p.load(fileName);
        int af = Integer.parseInt(p.getValue(new String[]{"Network"}, "Activation Function"));
        double Multiplyer = Double.parseDouble(p.getValue(new String[]{"Network"}, "Multiplier"));
        double Fitness = Double.parseDouble(p.getValue(new String[]{"Network"}, "fitness"));
        String sizes = p.getValue(new String[]{"Network"}, "sizes");
        int[] si = ParserTools.parseIntArray(sizes);
        GeneticNetwork ne = new GeneticNetwork(af, Multiplyer, si);
        ne.fitness = Fitness;
        
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
