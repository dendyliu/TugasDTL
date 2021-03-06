import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import java.util.Enumeration;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.Filter;

public class MyID3 extends Classifier {
  /** The node's successors. */ 
  private MyID3[] m_Successors;

  /** Attribute used for splitting. */
  private Attribute m_Attribute;

  /** Class value if node is leaf. */
  private double m_ClassValue;

  /** Class distribution if node is leaf. */
  private double[] m_Distribution;

  /** Class attribute of dataset. */
  private Attribute m_ClassAttribute;

  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();
    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);
    // instances
    result.setMinimumNumberInstances(0);
    return result;
  }

  public void buildClassifier(Instances data) throws Exception {
    // can classifier handle the data?
    getCapabilities().testWithFail(data);
    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();
    makeTree(data);
  }

  private void makeTree(Instances data) throws Exception {
    
    if(isOneClassOrEmptyAttribute(data)){
        makeLeaf(data);
    }else{
    	m_Attribute = chooseBestAttribute(data);
	    Instances[] splitData = splitData(data, m_Attribute);
	    m_Successors = new MyID3[m_Attribute.numValues()];
	    for (int j = 0; j < m_Attribute.numValues(); j++) {
	    	m_Successors[j] = new MyID3();
	        if(splitData[j].numInstances() <= 0){
		      	m_Successors[j].makeLeaf(data);
	      	} 
		  	else{
			    m_Successors[j].makeTree(splitData[j]);
		  	}
	    }
    }
  }

  private boolean isOneClassOrEmptyAttribute(Instances data) throws Exception{
    double[] infoGains = new double[data.numAttributes()];
    Enumeration attEnum = data.enumerateAttributes();
    while (attEnum.hasMoreElements()) {
      Attribute att = (Attribute) attEnum.nextElement();
      infoGains[att.index()]  = computeInfoGain(data, att);
    }
    m_Attribute = data.attribute(Utils.maxIndex(infoGains));
    if (Utils.eq(infoGains[m_Attribute.index()], 0))
      return true;
    return false;
  }
  //Cek if data example just just have one class distribution
  private boolean isOneClassExample(Instances data) throws Exception{
  	double[] classDistribution = new double[data.numClasses()];
  	Enumeration instEnum = data.enumerateInstances();
  	while (instEnum.hasMoreElements()) {
      Instance inst = (Instance) instEnum.nextElement();
      classDistribution[(int) inst.classValue()]++;
    }
    for(int i = 0; i < data.numClasses(); i++){
    	if(classDistribution[i]==data.numInstances())
    		return true;
    }
    return false;
  }
  //Procedure to create leaf 
  private void makeLeaf(Instances data) throws Exception{
  	m_Attribute = null;
    m_Distribution = new double[data.numClasses()];
    Enumeration instEnum = data.enumerateInstances();
    while (instEnum.hasMoreElements()) {
      Instance inst = (Instance) instEnum.nextElement();
      m_Distribution[(int) inst.classValue()]++;
    }
    Utils.normalize(m_Distribution);
    m_ClassValue = Utils.maxIndex(m_Distribution);
    m_ClassAttribute = data.classAttribute();
  }
  
  //Choose best attribute for node
  private Attribute chooseBestAttribute(Instances data) throws Exception{
  	// Compute attribute with maximum information gain.
    double[] infoGains = new double[data.numAttributes()];
    Enumeration attEnum = data.enumerateAttributes();
    while (attEnum.hasMoreElements()) {
      Attribute att = (Attribute) attEnum.nextElement();
      infoGains[att.index()]  = computeInfoGain(data, att);
    }
    return data.attribute(Utils.maxIndex(infoGains));
  }

  private double computeInfoGain(Instances data, Attribute att) throws Exception {
    double infoGain = computeEntropy(data);
    Instances[] splitData = splitData(data, att);
    for (int j = 0; j < att.numValues(); j++) {
      if (splitData[j].numInstances() > 0) {
        infoGain -= ((double) splitData[j].numInstances() /
                     (double) data.numInstances()) *
          computeEntropy(splitData[j]);
      }
    }
    return infoGain;
  }

  private double computeEntropy(Instances data) throws Exception {
    double [] classCounts = new double[data.numClasses()];
    Enumeration instEnum = data.enumerateInstances();
    while (instEnum.hasMoreElements()) {
      Instance inst = (Instance) instEnum.nextElement();
      classCounts[(int) inst.classValue()]++;
    }
    double entropy = 0;
    for (int j = 0; j < data.numClasses(); j++) {
      if (classCounts[j] > 0) {
        entropy -= classCounts[j] * Utils.log2(classCounts[j]);
      }
    }
    entropy /= (double) data.numInstances();
    return entropy + Utils.log2(data.numInstances());
  }

  private Instances[] splitData(Instances data, Attribute att) throws Exception{
  	Instances[] splitData = new Instances[att.numValues()];
  	for (int j = 0; j < att.numValues(); j++) {
		splitData[j] = new Instances(data, data.numInstances());
	}
	Enumeration instEnum = data.enumerateInstances();
	while (instEnum.hasMoreElements()) {
	   Instance inst = (Instance) instEnum.nextElement();
	   splitData[(int) inst.value(att)].add(inst);
	}
	for (int i = 0; i < splitData.length; i++) {
	    splitData[i].compactify();
	}
    return splitData;
  }
  //Delete attribute that become node
  private Instances removeAttributeNode(Instances oldData,int attIdx) throws Exception{
  	Instances newData = null;
  	try{
		Remove remover = new Remove();
	  	String[] optionsRemove = new String[2];
	  	optionsRemove[0] = "-R";
	  	optionsRemove[1] = String.valueOf(attIdx+1);
	  	remover.setOptions(optionsRemove);
		remover.setInputFormat(oldData);
		newData = Filter.useFilter(oldData,remover);
  	} catch(Exception e){
  		System.out.println("Exeption di removeAttribute "+e);
  	}
	return newData;
  }

  public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
    if (instance.hasMissingValue()) {
      throw new NoSupportForMissingValuesException("MyID3: no missing values, "+ "please.");
    }
    if (m_Attribute == null) {
      return m_ClassValue;
    } else {
      return m_Successors[(int) instance.value(m_Attribute)].
        classifyInstance(instance);
    }
  }

  public double[] distributionForInstance(Instance instance) throws NoSupportForMissingValuesException {
    if (instance.hasMissingValue()) {
      throw new NoSupportForMissingValuesException("Id3: no missing values, "+ "please.");
    }
    if (m_Attribute == null) {
      return m_Distribution;
    } else { 
      return m_Successors[(int) instance.value(m_Attribute)].
        distributionForInstance(instance);
    }
  }

  public Instances removeAtrInstance(Instance data,int attIdx){
  	Instances dataSet = new Instances(data.dataset());
  	dataSet.add(data);
  	Instances newDataSet = null;
  	try {
	  	Remove remover = new Remove();
		String[] optionsRemove = new String[2];
		optionsRemove[0] = "-R";
		optionsRemove[1] = String.valueOf(attIdx+1);
		remover.setOptions(optionsRemove);
	    remover.setInputFormat(dataSet);
	    newDataSet = Filter.useFilter(dataSet,remover);
	    newDataSet.compactify();
  	} catch(Exception e){
  		System.out.println(e);
  	}
  	return newDataSet;
  }

  public String toString(int level) {
    StringBuffer text = new StringBuffer();
    if (m_Attribute == null) {
      if (Instance.isMissingValue(m_ClassValue)) {
        text.append(": null");
      } else {
        text.append(": " + m_ClassAttribute.value((int) m_ClassValue));
      } 
    } else {
      for (int j = 0; j < m_Attribute.numValues(); j++) {
        text.append("\n");
        for (int i = 0; i < level; i++) {
          text.append("|  ");
        }
        text.append(m_Attribute.name() + " = " + m_Attribute.value(j));
        text.append(m_Successors[j].toString(level + 1));
      }
    }
    return text.toString();
  }
  
  public static void main(String[] args) {
    runClassifier(new MyID3(), args);
  }


}