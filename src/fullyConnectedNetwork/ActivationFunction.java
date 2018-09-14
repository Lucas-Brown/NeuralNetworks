package fullyConnectedNetwork;

public abstract class ActivationFunction {
	public short activationNum;
	
	abstract public double activator(double x);
	abstract public double derivative(double x);
	
	 public static class Linear extends ActivationFunction{ // (-inf, inf)
		 
		 public Linear() {
			 this.activationNum = 0;
		 }

			@Override
			public double activator(double x) {
				return x;
			}

			@Override
			public double derivative(double x) {
				return 1;
			}
			
	    }
	 
	 public static class BinaryStep extends ActivationFunction{ // {0,1}

		 public BinaryStep() {
			 this.activationNum = 1;
		 }
		 
			@Override
			public double activator(double x) {
				if(x < 0) {
					return 0;
				}else {
					return 1;
				}
			}

			@Override
			public double derivative(double x) {
				return 0;
			}
			
	    }
	 
	 public static class Sigmoid extends ActivationFunction{ // (0, 1)

		 public Sigmoid() {
			 this.activationNum = 2;
		 }
		 
			@Override
			public double activator(double x) {
				return 1.0 / (1.0 + Math.exp(-x));
			}

			@Override
			public double derivative(double x) {
				double i = this.activator(x);
				return (i) * (1.0 - i);
			}
			
	    }
	 
	 public static class TanH extends ActivationFunction{ // (-1, 1)

		 public TanH() {
			 this.activationNum = 3;
		 }
		 
			@Override
			public double activator(double x) {
				return Math.tanh(x);
			}

			@Override
			public double derivative(double x) {
				double i = this.activator(x);
				return 1 - i * i;
			}
			
	    }
	 
	 public static class ArcTan extends ActivationFunction{ // ( -pi/2, pi/2)

		 public ArcTan() {
			 this.activationNum = 4;
		 }
		 
			@Override
			public double activator(double x) {
				return Math.atan(x);
			}

			@Override
			public double derivative(double x) {
				return 1.0 / (x * x + 1.0);
			}
			
	    }
	 
	 public static class SoftSign extends ActivationFunction{ // (-1, 1)

		 public SoftSign() {
			 this.activationNum = 5;
		 }
		 
			@Override
			public double activator(double x) {
				return x / ( 1 + Math.abs(x));
			}

			@Override
			public double derivative(double x) {
				return 1.0 / Math.pow(1 + Math.abs(x), 2);
			}
			
	    }
	 
	 public static ActivationFunction intToActivationFunction(int i) {
		 if(i == 0) {
			 return new ActivationFunction.Linear();
		 }else if(i == 1) {
			 return new ActivationFunction.BinaryStep();
		 }else if(i == 2) {
			 return new ActivationFunction.Sigmoid();
		 }else if(i == 3) {
			 return new ActivationFunction.TanH();
		 }else if(i == 4) {
			 return new ActivationFunction.ArcTan();
		 }else if(i == 5) {
			 return new ActivationFunction.SoftSign();
		 }else {
			 return null;
		 }
	 }
	    
}
