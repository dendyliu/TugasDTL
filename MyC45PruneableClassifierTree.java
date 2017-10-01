import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.classifiers.trees.j48.*;

import java.util.ArrayList;


public class MyC45PruneableClassifierTree
        extends ClassifierTree {

    boolean m_pruneTheTree = false;

    float m_CF = 0.25f;

    private ArrayList<Rule> listOfRules;

    public MyC45PruneableClassifierTree(ModelSelection toSelectLocModel,
                                      boolean pruneTree,float cf)
            throws Exception {

        super(toSelectLocModel);
        m_pruneTheTree = pruneTree;
        m_CF = cf;
        listOfRules = new ArrayList<>();
    }

    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);
        data = new Instances(data);
        data.deleteWithMissingClass();
        buildTree(data, true);
        getRuleList(this, new Rule());
        int i = 0;
        for(Rule rule : listOfRules) {
            System.out.println("THIS IS RULE : " + i);
            rule.printPreConditions();
            i++;
        }
        if (m_pruneTheTree) {
            prune();
        }
    }

    public void prune() throws Exception {
//        double errorsLargestBranch;
//        double errorsLeaf;
//        double errorsTree;
//        int indexOfLargestBranch;
//        MyC45PruneableClassifierTree largestBranch;
//
//        if (!m_isLeaf){
//            for (int i = 0; i < m_sons.length; i++) {
//                son(i).prune();
//            }
//
//            indexOfLargestBranch = localModel().distribution().maxBag();
//            errorsLargestBranch = son(indexOfLargestBranch).
//                    getEstimatedErrorsForBranch((Instances)m_train);
//
//            errorsLeaf =
//                    getEstimatedErrorsForDistribution(localModel().distribution());
//
//            errorsTree = getEstimatedErrors();
//
//            if (Utils.smOrEq(errorsLeaf,errorsTree) && Utils.smOrEq(errorsLeaf,errorsLargestBranch)){
//                m_sons = null;
//                m_isLeaf = true;
//                m_localModel = new NoSplit(localModel().distribution());
//                return;
//            }
//
//            if (Utils.smOrEq(errorsLargestBranch,errorsTree+0.1)){
//                largestBranch = son(indexOfLargestBranch);
//                m_sons = largestBranch.m_sons;
//                m_localModel = largestBranch.localModel();
//                m_isLeaf = largestBranch.m_isLeaf;
//                creteNewDistribution(m_train);
//                prune();
//            }
//        }
    }

    private void getRuleList(MyC45PruneableClassifierTree node, Rule rule) {
        rule.addPreCondition(node);
        if (!node.m_isLeaf) {
            for (int i = 0; i < node.m_sons.length; i++) {
                getRuleList(node.son(i), new Rule(rule));
            }
        } else {
            listOfRules.add(rule);
        }
    }

    protected ClassifierTree getNewTree(Instances data) throws Exception {
        MyC45PruneableClassifierTree newTree =
                new MyC45PruneableClassifierTree(m_toSelectModel, m_pruneTheTree, m_CF);
        newTree.buildTree((Instances)data, true);

        return newTree;
    }

    private double getEstimatedErrors(){
        double errors = 0;
        int i;

        if (m_isLeaf)
            return getEstimatedErrorsForDistribution(localModel().distribution());
        else{
            for (i=0;i<m_sons.length;i++) {
                errors = errors + son(i).getEstimatedErrors();
            }
            return errors;
        }
    }

    private double getEstimatedErrorsForBranch(Instances data)
            throws Exception {
        Instances [] localInstances;
        double errors = 0;

        if (m_isLeaf) {
            return getEstimatedErrorsForDistribution(new Distribution(data));
        } else {
            Distribution savedDist = localModel().m_distribution;
            localModel().resetDistribution(data);
            localInstances = (Instances[])localModel().split(data);
            localModel().m_distribution = savedDist;
            for (int i = 0; i < m_sons.length; i++) {
                errors = errors +
                        son(i).getEstimatedErrorsForBranch(localInstances[i]);
            }
            return errors;
        }
    }

    private double getEstimatedErrorsForDistribution(Distribution
                                                             theDistribution){
        if (Utils.eq(theDistribution.total(),0)) {
            return 0;
        } else {
            return theDistribution.numIncorrect() +
                    Stats.addErrs(theDistribution.total(),
                            theDistribution.numIncorrect(), m_CF);
        }
    }

    private double getTrainingErrors(){
        double errors = 0;
        int i;
        if (m_isLeaf) {
            return localModel().distribution().numIncorrect();
        } else {
            for (i=0;i<m_sons.length;i++) {
                errors = errors + son(i).getTrainingErrors();
            }
            return errors;
        }
    }

    private ClassifierSplitModel localModel(){
        return (ClassifierSplitModel)m_localModel;
    }

    private void creteNewDistribution(Instances data) throws Exception {
        Instances [] localInstances;
        localModel().resetDistribution(data);
        m_train = data;
        if (!m_isLeaf){
            localInstances =
                    (Instances [])localModel().split(data);
            for (int i = 0; i < m_sons.length; i++) {
                son(i).creteNewDistribution(localInstances[i]);
            }
        } else {
            if (!Utils.eq(data.sumOfWeights(), 0)) {
                m_isEmpty = false;
            }
        }
    }

    private MyC45PruneableClassifierTree son(int index){
        return (MyC45PruneableClassifierTree)m_sons[index];
    }
}
