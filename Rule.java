import weka.classifiers.trees.j48.*;
import java.util.ArrayList;

public class Rule {
    private ArrayList<MyC45PruneableClassifierTree> preConditions;

    public Rule() {
        preConditions = new ArrayList<>();
    }

    public Rule(Rule rule) {
        preConditions = new ArrayList<>(rule.getPreConditions());
    }

    public void addPreCondition(MyC45PruneableClassifierTree node) {
        preConditions.add(node);
    }

    public ArrayList<MyC45PruneableClassifierTree> getPreConditions() {
        return preConditions;
    }

    public void printPreConditions() {
        for (MyC45PruneableClassifierTree tree : preConditions) {
            System.out.println(tree.toString());
        }
    }
}