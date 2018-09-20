package fullyConnectedNetwork;

import java.util.logging.Level;
import java.util.logging.Logger;

public abstract class GroupCalculate implements Cloneable {

	public abstract double[] calculate(Network[][] networkGroup, double... input);
	
	@Override
    public GroupCalculate clone(){
        try {
            return (GroupCalculate) super.clone();
        } catch (CloneNotSupportedException ex) {
            Logger.getLogger(GroupCalculate.class.getName()).log(Level.SEVERE, null, ex);
            return null;
        }
    }
}
