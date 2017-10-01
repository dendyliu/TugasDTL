import java.io.*;
import java.util.*;
import weka.classifiers.*;
import weka.classifiers.trees.*;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;
import weka.core.converters.ConverterUtils.DataSource;
public class WekaAccess {
//================================METHOD FOR DO THE SPECIFICATION TO ACCESS WEKA==============================================//
	public static Instances loadData (String file_data_name){
		Instances data = null;
		try{
			DataSource source = new DataSource("data\\seen\\"+file_data_name);
	 		data = source.getDataSet();
			if (data.classIndex() == -1)
			   data.setClassIndex(data.numAttributes() - 1);
			return data;
		} catch(Exception e){
			System.out.println(e);
		}
		return data;
	}
	public static Instances removeData(Instances oldData, String[] optionsRemove){
		Remove remover = new Remove();
		try {
			remover.setOptions(optionsRemove);
			remover.setInputFormat(oldData);
			Instances newData = Filter.useFilter(oldData,remover);
			System.out.println("Remove attribute . . . . . . . . . .");
			System.out.println("Instances:                  "+newData.numInstances());
			System.out.println("Attributes:                 "+newData.numAttributes());
			for(int i = 0; i < newData.numAttributes()	 - 1; i++){
		    System.out.println("Attribute Index-"+(i+1)+":          "+newData.attribute(i).name());
			}
			System.out.println("Attribute Class:            "+newData.classAttribute().name());
			System.out.println("Remove attribute done!\n");
			return newData;
		} catch(Exception e){
			System.out.println(e);
		}
		return oldData;
	}
	public static Instances resampleData(Instances oldData, String[] optionsResample){
		Resample resampler = new Resample();
		try {
			resampler.setBiasToUniformClass(1.0);
			resampler.setInputFormat(oldData);
			resampler.setNoReplacement(false);
			resampler.setRandomSeed(1);
			resampler.setSampleSizePercent(100);
			Instances newData = Filter.useFilter(oldData,resampler);
			return newData;
		} catch(Exception e){
			System.out.println(e);
		}
		return oldData;
	}

	public static Classifier assignClassifierModel(String[] optionsClassifier,String[] fn,String[] optionsSaveModel){
		Classifier classifier = null;
		try {
			if(Integer.parseInt(optionsClassifier[0]) == 1){
				classifier = new Id3();
				optionsSaveModel[1] = fn[0]+"_Id3.model";
			} else if(Integer.parseInt(optionsClassifier[0]) == 2){
				 String[] options = new String[1];
				 options[0] = "-Q";        
				 classifier = new J48();
				optionsSaveModel[1] = fn[0]+"_C45.model";
			} else if(Integer.parseInt(optionsClassifier[0]) == 3){
				classifier = new MyID3();
				optionsSaveModel[1] = fn[0]+"_MyID3.model";
			} else if(Integer.parseInt(optionsClassifier[0]) == 4){
				classifier = new MyC45();
				optionsSaveModel[1] = fn[0]+"_MyC45.model";
			}
		} catch(Exception e){
			System.out.println("Fail to build calssifier! " + e);
		}
		return classifier;
	}
	public static void printDTLModel(Classifier cls,String[] optionsClassifier) throws Exception{
		String model_string = "";
		if(Integer.parseInt(optionsClassifier[0]) == 1){
				Id3 model = (Id3) cls;
				model_string = model.toString();
			} else if(Integer.parseInt(optionsClassifier[0]) == 2){
			    J48 model = (J48) cls;
				model_string = model.toString();
			} else if(Integer.parseInt(optionsClassifier[0]) == 3){
				MyID3 model = (MyID3) cls;
				model_string = model.toString(0);
			} else if(Integer.parseInt(optionsClassifier[0]) == 4){
				MyC45 model = (MyC45) cls;
				model_string = model.toString();
			}
		System.out.println("Model Tree: \n" +model_string +"\n");	
	}
	public static void doEvalAndClassify(Instances data, String[] optionsEvaluation, Classifier cls,String[] optionsClassifier) throws Exception{
		System.out.println("Building your classifier . . . . . . . . . . . . .");
		Evaluation eval = null;
		if(Integer.parseInt(optionsEvaluation[0]) == 1){
			eval = new Evaluation(data);
			eval.crossValidateModel(cls,data,Integer.parseInt(optionsEvaluation[1]),new Random(1));
		} else if(Integer.parseInt(optionsEvaluation[0]) == 2){
			data.randomize(new java.util.Random(1));
			int trainSize = (int) Math.round(data.numInstances() * Double.parseDouble(optionsEvaluation[1])/100);
			int testSize = data.numInstances() - trainSize;
			Instances train_data = new Instances(data, 0, trainSize);
			Instances test_data = new Instances(data, trainSize, testSize);
			cls.buildClassifier(train_data);
			eval = new Evaluation(train_data);
			eval.evaluateModel(cls,test_data);
		}else if(Integer.parseInt(optionsEvaluation[0]) == 3){
			Instances test_data = new Instances(new BufferedReader(new FileReader("data\\test\\"+optionsEvaluation[1])));
			cls.buildClassifier(data);
			test_data.setClassIndex(data.numAttributes() - 1);
			eval = new Evaluation(data);
			eval.evaluateModel(cls,test_data);
		}

		System.out.print(eval.toSummaryString("\nEvaluation Results\n===================================================================", false));
		System.out.println("===================================================================");
		System.out.println("Building classifier finish!");
	}

	public static void doClassifyUnseenData(String[] optionsClassify, Classifier cls) throws IOException,Exception{
		System.out.println("\nClassify your unseen data . . . . . . . . . . . . .\n");
		Instances unlabeled = new Instances(new BufferedReader(new FileReader("data\\unseen\\"+optionsClassify[1])));
		unlabeled.setClassIndex(unlabeled.numAttributes() - 1);
		Instances labeled = new Instances(unlabeled);
		 for (int i = 0; i < unlabeled.numInstances(); i++) {
		   double clsLabel = cls.classifyInstance(unlabeled.instance(i));
		   labeled.instance(i).setClassValue(clsLabel);
		 }
		 BufferedWriter writer = new BufferedWriter(new FileWriter("data\\result_unseen\\classified_"+optionsClassify[1]));
		 writer.write(labeled.toString());
		 writer.newLine();
		 writer.flush();
		 writer.close();
		 System.out.println("Proccess classify unseen data finish!");
		 System.out.println("Result file in data\\result_unseen\\classified_"+optionsClassify[1]);
	}

	public static void doEvalFromLoadModel(Instances data, String[] optionsLoadModel,
												String[] optionsEvaluation,Classifier cls) throws Exception{
		System.out.println("Loading your classifier . . . . . . . . . . . . .");
		cls = (Classifier) SerializationHelper.read("data\\model\\"+optionsLoadModel[1]);
		Evaluation eval = null;
		if(Integer.parseInt(optionsEvaluation[0]) == 1){
			eval = new Evaluation(data);
			eval.crossValidateModel(cls,data,Integer.parseInt(optionsEvaluation[1]),new Random(1));
		} else if(Integer.parseInt(optionsEvaluation[0]) == 2){
			data.randomize(new java.util.Random(1));
			int trainSize = (int) Math.round(data.numInstances() * Double.parseDouble(optionsEvaluation[1])/100);
			int testSize = data.numInstances() - trainSize;
			Instances train_data = new Instances(data, 0, trainSize);
			Instances test_data = new Instances(data, trainSize, testSize);
			eval = new Evaluation(train_data);
			eval.evaluateModel(cls,test_data);
		}else if(Integer.parseInt(optionsEvaluation[0]) == 3){
			Instances test_data = new Instances(new BufferedReader(new FileReader("data\\test\\"+optionsEvaluation[1])));
			test_data.setClassIndex(data.numAttributes() - 1);
			eval = new Evaluation(data);
			eval.evaluateModel(cls,test_data);
		}
		System.out.print(eval.toSummaryString("\nEvaluation Results\n===================================================================", false));
		System.out.println("===================================================================");
		System.out.println("Load classifier finish!");
	}
//============================================================================================================================//
//================================METHOD FOR GET OPTIONS MENU=================================================================//
	public static void printAllDataFile(String[] fn){
		Scanner scan =  new Scanner(System.in);
		File folder = new File("data\\seen");
		File[] listOfFiles = folder.listFiles();
	    for (int i = 0; i < listOfFiles.length; i++) {
	      if (listOfFiles[i].isFile()) {
	        System.out.println("  -Data "+(i+1)+": " + listOfFiles[i].getName());
	      } 
    	}
    	System.out.print("   Your choose: ");
    	int a = scan.nextInt();
    	fn[0] = listOfFiles[a-1].getName().toString();
	}

	public static void isRemoveAttribute(String[] optionsRemove){
		Scanner scan =  new Scanner(System.in);
		String isRemove = scan.nextLine();
		if(isRemove.toLowerCase().charAt(0)=='y'){
			optionsRemove[0] = "-R";
			System.out.print("   a.Index attribute range to be removed : ");
			int firstIdxAtr = scan.nextInt();
			optionsRemove[1]  = String.valueOf(firstIdxAtr);
		} else if(isRemove.toLowerCase().charAt(0)=='n'){
			return;
		} else {
			System.out.println("Wrong command!");
			isRemoveAttribute(optionsRemove);
		}
	}

	public static void isResampleData(String[] optionsResample){
		Scanner scan = new Scanner(System.in);
		String isResample = scan.nextLine();
		if(isResample.toLowerCase().charAt(0)=='y'){
			//OPTIONS NYA MASIH DI TESTING
			optionsResample[0] = "B";
		} else if(isResample.toLowerCase().charAt(0)=='n'){
			return;
		} else {
			System.out.println("Wrong command!");
			isResampleData(optionsResample);
		}
	}

	public static void printAllClassifierOption(String[] optionsClassifier){
		Scanner scan = new Scanner(System.in);
		System.out.println("   1.ID3 weka");
		System.out.println("   2.C45 weka");
		System.out.println("   3.myID3");
		System.out.println("   4.myC45");
		System.out.print("   Your choose: ");
		optionsClassifier[0] = String.valueOf(scan.nextInt());
	}

	public static void printAllEvaluationOption(String[] optionsEvaluation){
		Scanner scan = new Scanner(System.in);
		System.out.println("   1.Cross Validation");
		System.out.println("   2.Split Percentage");
		System.out.println("   3.Training-Testing");
		System.out.print("   Your choose: ");
		optionsEvaluation[0] = String.valueOf(scan.nextInt());
		if(Integer.parseInt(optionsEvaluation[0]) == 1){
			System.out.print("   Fold number: ");
			optionsEvaluation[1] = String.valueOf(scan.nextInt());
		}else if(Integer.parseInt(optionsEvaluation[0]) == 2){
			System.out.print("   Percentage number: ");
			optionsEvaluation[1] = String.valueOf(scan.nextInt());
		}else if(Integer.parseInt(optionsEvaluation[0]) == 3){
			System.out.println("   Testing data name: ");
			File folder = new File("data\\test");
			File[] listOfFiles = folder.listFiles();
		    for (int i = 0; i < listOfFiles.length; i++) {
		      if (listOfFiles[i].isFile()) {
		        System.out.println("  -Data "+(i+1)+": " + listOfFiles[i].getName());
		      } 
	    	}
	    	System.out.print("   Your choose: ");
	    	int a = scan.nextInt();
	    	optionsEvaluation[1]= listOfFiles[a-1].getName().toString();
		}
	}
	public static void isSaveModel(String[] optionsSaveModel){
		Scanner scan = new Scanner(System.in);
		String isSaveModel = scan.nextLine();
		if(isSaveModel .toLowerCase().charAt(0)=='y'){
			optionsSaveModel[0] = "Y";
		}else if(isSaveModel.toLowerCase().charAt(0)=='n'){
			return;
		} else {
			System.out.println("Wrong command!");
			isSaveModel(optionsSaveModel);
		}
	}
	public static void isClassifyUnseenData(String[] optionsClassify){
		Scanner scan = new Scanner(System.in);
		String isClassify = scan.nextLine();
		if(isClassify.toLowerCase().charAt(0)=='y'){
			File folder = new File("data\\unseen");
			File[] listOfFiles = folder.listFiles();
		    for (int i = 0; i < listOfFiles.length; i++) {
		      if (listOfFiles[i].isFile()) {
		        System.out.println("  -Data "+(i+1)+": " + listOfFiles[i].getName());
		      } 
	    	}
	    	System.out.print("   Your choose: ");
	    	int a = scan.nextInt();
	    	optionsClassify[0] = "Y";
	    	optionsClassify[1] = listOfFiles[a-1].getName().toString();
		}else if(isClassify.toLowerCase().charAt(0)=='n'){
			return;
		} else {
			System.out.println("Wrong command!");
			isClassifyUnseenData(optionsClassify);
		}
	}

	public static void isLoadModel(String[] optionsLoadModel){
		Scanner scan = new Scanner(System.in);
		String isLoad = scan.nextLine();
		if(isLoad.toLowerCase().charAt(0)=='y'){
			System.out.println("");
			System.out.println("  Choose your model:");
			File folder = new File("data\\model");
			File[] listOfFiles = folder.listFiles();
		    for (int i = 0; i < listOfFiles.length; i++) {
		      if (listOfFiles[i].isFile()) {
		        System.out.println("  -Data "+(i+1)+": " + listOfFiles[i].getName());
		      } 
	    	}
	    	System.out.print("   Your choose: ");
	    	int a = scan.nextInt();
	    	optionsLoadModel[0] = "Y";
	    	optionsLoadModel[1] = listOfFiles[a-1].getName().toString();
		}else if(isLoad.toLowerCase().charAt(0)=='n'){
			return;
		} else {
			System.out.println("Wrong command!");
			isLoadModel(optionsLoadModel);
		}
	}
//============================================================================================================================//
	public static void runMenu(String[] fn, String[] optionsRemove, String[] optionsResample, 
								String[] optionsClassifier,String[] optionsEvaluation,String[] optionsSaveModel,
								String[] optionsClassify, String[] optionsLoadModel){
		System.out.println("==================================================");
		System.out.println("       Tubes 1B IMPLEMENTING WEKA IN JAVA ");
		System.out.println("==================================================");
		System.out.println("");
		System.out.println("1.Choose your data:");
		printAllDataFile(fn);
		System.out.println("");
		System.out.print("2.Remove Attribute? (Y/N) ");
		isRemoveAttribute(optionsRemove);
		System.out.println("");
		System.out.print("3.Using Resample? (Y/N) ");
		isResampleData(optionsResample);
		System.out.println("");
		System.out.print("4.Load model from file? (Y/N) ");
		isLoadModel(optionsLoadModel);
		if(optionsLoadModel[0]=="Y"){
			System.out.println("");
			System.out.println("5.Choose your evaluation type:");
			printAllEvaluationOption(optionsEvaluation);
			System.out.println("");
			System.out.print("6.Want to classify unseen data? (Y/N) ");
			isClassifyUnseenData(optionsClassify);
		}else{
		  	System.out.println("");
			System.out.println("5.Choose your classifier:");
			printAllClassifierOption(optionsClassifier);
			System.out.println("");
			System.out.println("6.Choose your evaluation type:");
			printAllEvaluationOption(optionsEvaluation);
			System.out.println("");
			System.out.print("7.Do you want to save model? (Y/N) ");
			isSaveModel(optionsSaveModel);
			System.out.println("");
			System.out.print("8.Want to classify unseen data? (Y/N) ");
			isClassifyUnseenData(optionsClassify);
		}
		
	}


	public static void main (String args[]){
		String[] fn= new String[2];
		String[] optionsRemove = new String[2];
		String[] optionsResample = new String[10]; //Tes resample
		String[] optionsClassifier = new String[2];
		String[] optionsEvaluation = new String[2];
		String[] optionsClassify = new String[2];
		String[] optionsSaveModel = new String[2];
		String[] optionsLoadModel = new String[2];
		System.out.println("");
		runMenu(fn,optionsRemove,optionsResample,optionsClassifier,optionsEvaluation,optionsSaveModel
			,optionsClassify,optionsLoadModel);
		Instances data;
		try {
			data = new Instances(loadData(fn[0]));
			System.out.println("\nData Information\n===================================================================");
			System.out.println("Relation:                   "+data.relationName());
			System.out.println("Instances:                  "+data.numInstances());
			System.out.println("Attributes:                 "+data.numAttributes());
			for(int i = 0; i < data.numAttributes()	 - 1; i++){
		    System.out.println("Attribute Index-"+(i+1)+":          "+data.attribute(i).name());
			}
			System.out.println("Attribute Class:            "+data.classAttribute().name());
			System.out.println("===================================================================");
			System.out.println("");
			if(optionsRemove[0]=="-R")
				data = removeData(data,optionsRemove);
			if(optionsResample[0]=="B")
				data = resampleData(data,optionsResample);
			Classifier yourClassifier = null;
			if(optionsLoadModel[0]=="Y"){
				doEvalFromLoadModel(data,optionsLoadModel,optionsEvaluation,yourClassifier);
			} else{
				yourClassifier = assignClassifierModel(optionsClassifier,fn,optionsSaveModel);
				doEvalAndClassify(data,optionsEvaluation,yourClassifier,optionsClassifier);
				if(optionsSaveModel[0]=="Y")
					SerializationHelper.write("data\\model\\"+optionsSaveModel[1], yourClassifier);
			}
		    if(optionsClassify[0]=="Y")
		    	doClassifyUnseenData(optionsClassify,yourClassifier);

		} catch(IOException ioe){
			System.out.println("No file with that name in data folder!"+ioe);
		} catch(Exception e){
			System.out.println(e);
		}
	}
}