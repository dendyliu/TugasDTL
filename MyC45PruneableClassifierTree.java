import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.classifiers.trees.j48.*;


public class MyC45PruneableClassifierTree
        extends ClassifierTree {

    boolean m_pruneTheTree = false;

    float m_CF = 0.25f;

    boolean m_subtreeRaising = true;

    boolean m_cleanup = true;

    public MyC45PruneableClassifierTree(ModelSelection toSelectLocModel,
                                      boolean pruneTree,float cf,
                                      boolean raiseTree,
                                      boolean cleanup)
            throws Exception {

        super(toSelectLocModel);

        m_pruneTheTree = pruneTree;
        m_CF = cf;
        m_subtreeRaising = raiseTree;
        m_cleanup = cleanup;
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
        // can classifier tree handle the data?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        buildTree(data, m_subtreeRaising || !m_cleanup);
        collapse();
        if (m_pruneTheTree) {
            prune();
        }
        if (m_cleanup) {
            cleanup(new Instances(data, 0));
        }
    }

    public final void collapse(){

        double errorsOfSubtree;
        double errorsOfTree;
        int i;

        if (!m_isLeaf){
            errorsOfSubtree = getTrainingErrors();
            errorsOfTree = localModel().distribution().numIncorrect();
            if (errorsOfSubtree >= errorsOfTree-1E-3){

                // Free adjacent trees
                m_sons = null;
                m_isLeaf = true;

                // Get NoSplit Model for tree.
                m_localModel = new NoSplit(localModel().distribution());
            }else
                for (i=0;i<m_sons.length;i++)
                    son(i).collapse();
        }
    }

    public void prune() throws Exception {
        double errorsLargestBranch;
        double errorsLeaf;
        double errorsTree;
        int indexOfLargestBranch;
        MyC45PruneableClassifierTree largestBranch;
        int i;

        if (!m_isLeaf){

            // Prune all subtrees.
            for (i=0;i<m_sons.length;i++)
                son(i).prune();

            // Compute error for largest branch
            indexOfLargestBranch = localModel().distribution().maxBag();
            if (m_subtreeRaising) {
                errorsLargestBranch = son(indexOfLargestBranch).
                        getEstimatedErrorsForBranch((Instances)m_train);
            } else {
                errorsLargestBranch = Double.MAX_VALUE;
            }

            // Compute error if this Tree would be leaf
            errorsLeaf =
                    getEstimatedErrorsForDistribution(localModel().distribution());

            // Compute error for the whole subtree
            errorsTree = getEstimatedErrors();

            // Decide if leaf is best choice.
            if (Utils.smOrEq(errorsLeaf,errorsTree+0.1) &&
                    Utils.smOrEq(errorsLeaf,errorsLargestBranch+0.1)){

                // Free son Trees
                m_sons = null;
                m_isLeaf = true;

                // Get NoSplit Model for node.
                m_localModel = new NoSplit(localModel().distribution());
                return;
            }

            // Decide if largest branch is better choice
            // than whole subtree.
            if (Utils.smOrEq(errorsLargestBranch,errorsTree+0.1)){
                largestBranch = son(indexOfLargestBranch);
                m_sons = largestBranch.m_sons;
                m_localModel = largestBranch.localModel();
                m_isLeaf = largestBranch.m_isLeaf;
                newDistribution(m_train);
                prune();
            }
        }
    }

    protected ClassifierTree getNewTree(Instances data) throws Exception {
        MyC45PruneableClassifierTree newTree =
                new MyC45PruneableClassifierTree(m_toSelectModel, m_pruneTheTree, m_CF,
                        m_subtreeRaising, m_cleanup);
        newTree.buildTree((Instances)data, m_subtreeRaising || !m_cleanup);

        return newTree;
    }

    private double getEstimatedErrors(){
        double errors = 0;
        int i;

        if (m_isLeaf)
            return getEstimatedErrorsForDistribution(localModel().distribution());
        else{
            for (i=0;i<m_sons.length;i++)
                errors = errors+son(i).getEstimatedErrors();
            return errors;
        }
    }

    private double getEstimatedErrorsForBranch(Instances data)
            throws Exception {
        Instances [] localInstances;
        double errors = 0;
        int i;

        if (m_isLeaf)
            return getEstimatedErrorsForDistribution(new Distribution(data));
        else{
            Distribution savedDist = localModel().m_distribution;
            localModel().resetDistribution(data);
            localInstances = (Instances[])localModel().split(data);
            localModel().m_distribution = savedDist;
            for (i=0;i<m_sons.length;i++)
                errors = errors+
                        son(i).getEstimatedErrorsForBranch(localInstances[i]);
            return errors;
        }
    }

    private double getEstimatedErrorsForDistribution(Distribution
                                                             theDistribution){
        if (Utils.eq(theDistribution.total(),0))
            return 0;
        else
            return theDistribution.numIncorrect()+
                    Stats.addErrs(theDistribution.total(),
                            theDistribution.numIncorrect(),m_CF);
    }

    private double getTrainingErrors(){

        double errors = 0;
        int i;

        if (m_isLeaf)
            return localModel().distribution().numIncorrect();
        else{
            for (i=0;i<m_sons.length;i++)
                errors = errors+son(i).getTrainingErrors();
            return errors;
        }
    }

    private ClassifierSplitModel localModel(){
        return (ClassifierSplitModel)m_localModel;
    }

    private void newDistribution(Instances data) throws Exception {
        Instances [] localInstances;

        localModel().resetDistribution(data);
        m_train = data;
        if (!m_isLeaf){
            localInstances =
                    (Instances [])localModel().split(data);
            for (int i = 0; i < m_sons.length; i++)
                son(i).newDistribution(localInstances[i]);
        } else {

            // Check whether there are some instances at the leaf now!
            if (!Utils.eq(data.sumOfWeights(), 0)) {
                m_isEmpty = false;
            }
        }
    }

    private MyC45PruneableClassifierTree son(int index){
        return (MyC45PruneableClassifierTree)m_sons[index];
    }
}
