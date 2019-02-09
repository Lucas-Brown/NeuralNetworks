package genetics;

import java.util.ArrayList;

public class IntArrList{
    private ArrayList<int[]> aList = new ArrayList<int[]>();

    public int[] get(int index){
        return this.aList.get(index);
    }

    public void add(int[] e){
        this.aList.add(e);
    }

    public int size(){
        return this.aList.size();
    }
}