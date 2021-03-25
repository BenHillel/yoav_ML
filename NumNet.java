import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;

public class NumNet {
	
	String imagesMats_path = "C:\\Users\\user\\Documents\\Python\\image staff\\imagesMats.txt";
	String imagesVals_path = "C:\\Users\\user\\Documents\\Python\\image staff\\imagesVals.txt";
	
	double[][] trainingData;
	int[] trainingValues;
	int examples = getLength(imagesVals_path);
	int secceses;
	int trys;
	int maxSecceses;
	
	double[] input;
	double[] lair1;
	double[] lair2;
	double[] output;
	double[][] weights1;
	double[] biase1;
	double[][] weights2;
	double[] biase2;
	double[][] weights3;
	double[] biase3;
	int inputSize = 3600;
	int expValue;
	

	
	public NumNet(int len1,int len2) {
		this.input = new double[inputSize];
		this.lair1 = new double[len1];
		this.lair2 = new double[len2];
		this.output = new double[10];
		this.weights1 = new double[len1][inputSize];
		this.biase1 = new double[len1];
		this.weights2 = new double[len2][len1];
		this.biase2 = new double[len2];
		this.weights3 = new double[10][len2];
		this.biase3 = new double[10];
		
		this.trainingData = new double[examples][inputSize];
		this.trainingValues = new int[examples];
		this.secceses=0;
		this.trys=1;
		this.maxSecceses=0;
		
	}
	
	public void randomSetUp() {
		int len1 = this.lair1.length;
		int len2 = this.lair2.length;
		Random r = new Random();
		
		for(int t=0;t<this.examples;t++) {
			for(int i=0;i<inputSize;i++) {
				this.trainingData[t][i] = Math.random();
			}
			this.trainingValues[t] = r.nextInt(10);
		}
		
		//input setup ------------------------------------
		for(int i=0;i<inputSize;i++) {
			this.input[i] = Math.random();
		}
		
		
		//biases and weights setup -----------------------
		for(int i=0;i<len1;i++) {
			this.biase1[i] = (Math.random()*10)-5;
			for(int j=0;j<inputSize;j++) {
				weights1[i][j] = (Math.random()*4)-2;
			}
		}
		for(int i=0;i<len2;i++) {
			this.biase2[i] = (Math.random()*10)-5;
			for(int j=0;j<len1;j++) {
				weights2[i][j] = (Math.random()*4)-2;
			}
		}
		for(int i=0;i<10;i++) {
			this.biase3[i] = (Math.random()*10)-5;
			for(int j=0;j<len2;j++) {
				weights3[i][j] = (Math.random()*4)-2;
			}
		}
		
	}
	
	public void NotRandomSetUp() {
		int len1 = this.lair1.length;
		int len2 = this.lair2.length;
		
		
		this.trainingData = readData(imagesMats_path,examples,inputSize);
		
		this.trainingValues = readDataInt(imagesVals_path, examples);
		
		//biases and weights setup -----------------------
		this.biase1 = readDataDouble("biase1.txt",len1);
		this.biase2 = readDataDouble("biase2.txt",len2);
		this.biase3 = readDataDouble("biase3.txt",10);
		
		this.weights1 = readData("weights1.txt",len1,inputSize);
		this.weights2 = readData("weights2.txt",len2,len1);
		this.weights3 = readData("weights3.txt",10,len2);
		
	}

	
	public double sigmoid(double num) {
		return 1/(1+Math.pow(Math.E,-num));
	}
	public double[] sigmoid(double[] vec) {
		for(int i=0;i<vec.length;i++) {
			vec[i] = sigmoid(vec[i]);
		}
		return vec;
	}
	public double[] multiply(double[][] mat,double[] vec) {
		double[] result = new double[mat.length];
		for(int i=0;i<mat.length;i++) {
			for(int j=0;j<mat[i].length;j++) {
				result[i] += mat[i][j]*vec[j];
			}
		}
		return result;
	}
	public double[] add(double[] vec1,double[] vec2) {
		double[] result = new double[vec1.length];
		for(int i=0;i<vec1.length;i++) {
			result[i] = vec1[i]+vec2[i];
		}
		return result;
	}
	public int maxIndex(double[] vec) {
		double max = 0;
		int index = 0;
		for(int i=0;i<vec.length;i++) {
			if(vec[i]>max) {
				max = vec[i];
				index = i;
			}
		}
		return index;
	}
	public int max(int x, int y) {
		if(x>=y) {
			return x;
		}
		return y;
	}
	public int min(int x, int y) {
		if(x<=y) {
			return x;
		}
		return y;
	}
	
	public int calculate() {
		this.lair1 = sigmoid(add(multiply(this.weights1,this.input),this.biase1));
		this.lair2 = sigmoid(add(multiply(this.weights2,this.lair1),this.biase2));
		this.output = sigmoid(add(multiply(this.weights3,this.lair2),this.biase3));
		return maxIndex(this.output);
	}
	
	public double cost() {
		double cost = 0;
		this.calculate();
		int expectedValue = this.expValue;
		for(int i=0;i<10;i++) {
			if(i==expectedValue) {
				cost += Math.pow((this.output[i]-1), 2);
			}else {
				cost += this.output[i]*this.output[i];
			}
		}
		return cost;
	}
	
	public void linearRegretion(double alpha) {
		double[] expectedValue = new double[10];
		for(int i=0;i<10;i++) {
			if(i==this.expValue) {
				expectedValue[i] = 1;
			}else {
				expectedValue[i] = 0;
			}
		}
		//first lair regretion --------------------------------
		for(int i=0;i<this.weights3.length;i++) {
			for(int j=0;j<this.weights3[i].length;j++) {
				this.weights3[i][j] -= alpha*2*lair2[j]*(this.output[i]-expectedValue[i]);
			}
			this.biase3[i] -= alpha*2*(this.output[i]-expectedValue[i]);
		}
		//second lair regration -------------------------------
		for(int i=0;i<this.weights2.length;i++) {
			double pred = this.output[i];
			double y = expectedValue[i];
			for(int j=0;j<this.weights2[i].length;j++) {
				double a = this.lair1[j];
				for(int k=0;k<10;k++) {
					this.weights2[i][j] -= 2*alpha*this.weights3[k][i]*a*(pred-y);
					this.biase2[i] -= 2*alpha*this.weights3[k][i]*(pred-y);
				}
				
			}
		}
		//third lair regration ---------------------------------
		for(int i=0;i<this.weights1.length;i++) {
			double pred = this.output[i];
			double y = expectedValue[i];
			for(int j=0;j<this.weights1[i].length;j++) {
				double input = this.input[j];
				for(int k=0;k<10;k++) {
					for(int l=0;l<this.lair2.length;l++) {
						this.weights1[i][j] -= 2*alpha*this.weights3[k][l]*this.weights2[l][i]*input*(pred-y);
						this.biase1[i] -= 2*alpha*this.weights3[k][l]*this.weights2[l][i]*(pred-y);
					}
				}
			}
		}
	}
	
	public void train(int index,double alpha,boolean show) {
		this.input = this.trainingData[index];
		this.expValue = this.trainingValues[index];
		String didIt = "X";
		if(this.expValue==this.calculate()) {
			this.secceses++;
			didIt = "V";
		}
		this.trys++;
		if(show) {
			if (this.maxSecceses < this.secceses) {
				this.maxSecceses = this.secceses;
				this.updateData();
				System.out.println("---DATA_SAVED---");
			}
			
			System.out.println(this.expValue+"  |  "+this.calculate()+"   "+didIt+"   seccess rate = "+(100*this.secceses/(double)this.trys)+'%');
			//System.out.println(this.cost());
			this.trys = 0;
			this.secceses = 0;
		}
		this.linearRegretion(alpha);
	}
	
	public void updateData() {
		try {
			File weights1 = new File("weights1.txt");
			if(weights1.createNewFile()) {
				//System.out.println("weights1 created");
			}else {
				//System.out.println("weights1 already exist");
			}
			FileWriter weights1W = new FileWriter("weights1.txt");
			for(int i=0;i<this.weights1.length;i++) {
				for(int j=0;j<this.weights1[i].length;j++) {
					if(!(i==j && i ==0)) {
						weights1W.write("\n");
					}
					String val = String.valueOf(this.weights1[i][j]);
					weights1W.write(val);
				} 
			}
			weights1W.close();
			//-------------------------------------------------------
			File weights2 = new File("weights2.txt");
			if(weights2.createNewFile()) {
				//System.out.println("weights2 created");
			}else {
				//System.out.println("weights2 already exist");
			}
			FileWriter weights2W = new FileWriter("weights2.txt");
			for(int i=0;i<this.weights2.length;i++) {
				for(int j=0;j<this.weights2[i].length;j++) {
					if(!(i==j && i ==0)) {
						weights2W.write("\n");
					}
					String val = String.valueOf(this.weights2[i][j]);
					weights2W.write(val);
				} 
			}
			weights2W.close();
			//-------------------------------------------------------
			File weights3 = new File("weights3.txt");
			if(weights3.createNewFile()) {
				//System.out.println("weights3 created");
			}else {
				//System.out.println("weights3 already exist");
			}
			FileWriter weights3W = new FileWriter("weights3.txt");
			for(int i=0;i<this.weights3.length;i++) {
				for(int j=0;j<this.weights3[i].length;j++) {
					if(!(i==j && i ==0)) {
						weights3W.write("\n");
					}
					String val = String.valueOf(this.weights3[i][j]);
					weights3W.write(val);
				} 
			}
			weights3W.close();
			//-------------------------------------------------------
			File biase1 = new File("biase1.txt");
			if(biase1.createNewFile()) {
				//System.out.println("biase1 created");
			}else {
				//System.out.println("biase1 already exist");
			}
			FileWriter biase1W = new FileWriter("biase1.txt");
			for(int i=0;i<this.biase1.length;i++) {
				if(i != 0) {
					biase1W.write("\n");
				}
				String val = String.valueOf(this.biase1[i]);
				biase1W.write(val);
			}
			biase1W.close();
			//-------------------------------------------------------
			File biase2 = new File("biase2.txt");
			if(biase2.createNewFile()) {
				//System.out.println("biase2 created");
			}else {
				//System.out.println("biase2 already exist");
			}
			FileWriter biase2W = new FileWriter("biase2.txt");
			for(int i=0;i<this.biase2.length;i++) {
				if(i != 0) {
					biase2W.write("\n");
				}
				String val = String.valueOf(this.biase2[i]);
				biase2W.write(val);
			}
			biase2W.close();
			//-------------------------------------------------------
			File biase3 = new File("biase3.txt");
			if(biase3.createNewFile()) {
				//System.out.println("biase3 created");
			}else {
				//System.out.println("biase3 already exist");
			}
			FileWriter biase3W = new FileWriter("biase3.txt");
			for(int i=0;i<this.biase3.length;i++) {
				if(i != 0) {
					biase3W.write("\n");
				}
				String val = String.valueOf(this.biase3[i]);
				biase3W.write(val);
			}
			biase3W.close();
		}catch(IOException e) {
			
		}
	}
	
	public double[][] readData(String path,int width,int height) {
		//First line of the input data should be how many lines does it have
		double[][] data= new double[width][height];
		try {
			File file = new File(path);
			Scanner reader = new Scanner(file);
			int i = 0;
			int j = 0;
			while (reader.hasNext()) {
				data[j][i] = reader.nextDouble();
				i++;
				if(i == height) {
					j++;
					i = 0;
				}
			}
			reader.close();
		}catch(FileNotFoundException e) {
			e.printStackTrace();
		}
		return data;
	}
	
	public int[] readDataInt(String path,int length) {
		//First line of the input data should be how many lines does it have
		int[] data= new int[length];
		try {
			File file = new File(path);
			Scanner reader = new Scanner(file);
			int i = 0;
			while (reader.hasNext()) {
				data[i] = reader.nextInt();
				i++;
			}
			reader.close();
		}catch(FileNotFoundException e) {
			e.printStackTrace();
		}
		return data;
	}
	
	public double[] readDataDouble(String path,int length) {
		//First line of the input data should be how many lines does it have
		double[] data= new double[length];
		try {
			File file = new File(path);
			Scanner reader = new Scanner(file);
			int i = 0;
			while (reader.hasNext()) {
				data[i] = reader.nextDouble();
				i++;
			}
			reader.close();
		}catch(FileNotFoundException e) {
			e.printStackTrace();
		}
		return data;
	}
	
	public int getLength(String path) {
		int i = 0;
		try {
			File file = new File(path);
			Scanner reader = new Scanner(file);
			while (reader.hasNext()) {
				i++;
				reader.nextInt();
			}
			reader.close();
		}catch(FileNotFoundException e) {
			e.printStackTrace();
		}
		System.out.println(i);
		return i;
	}
	
	public static void main(String[] args) {
		NumNet net = new NumNet(5,5);
		net.NotRandomSetUp();
		Random r = new Random();
		int i = 0;
		while(true) {
			i++;
			if(i%1000==0) {
				net.train(r.nextInt(net.examples), 0.1,true);
				//net.updateData();
			}else {
				net.train(r.nextInt(net.examples), 0.1,false);
			}
		}
		
	}
}
