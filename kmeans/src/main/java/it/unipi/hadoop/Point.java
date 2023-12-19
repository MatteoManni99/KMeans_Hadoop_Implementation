package it.unipi.hadoop;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class Point implements Writable {

    private FloatWritable[] features;
    private final IntWritable dimension;
    private final IntWritable sumCounter;

    //constructor for a Point object computed from scratch
    public Point () {
        dimension = new IntWritable(0);
        sumCounter = new IntWritable(0);
        features = new FloatWritable[0];
    }

    //constructor for a Point object retrieved from a String
    public Point (String value){
        String[] tokens = value.trim().split(",");
        sumCounter = new IntWritable(1);
        dimension = new IntWritable(tokens.length);
        features = new FloatWritable[dimension.get()];

        for (int i = 0; i < dimension.get(); i++)
            features[i] = new FloatWritable(Float.parseFloat(tokens[i]));
    }

    //'write' and 'readFields' for serialization
    @Override
    public void write(DataOutput dataOutput) throws IOException {
        dataOutput.writeInt(dimension.get());
        dataOutput.writeInt(sumCounter.get());

        for (FloatWritable floatValue : features) {
            dataOutput.writeFloat(floatValue.get());
        }
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        dimension.set(dataInput.readInt());
        sumCounter.set(dataInput.readInt());

        features = new FloatWritable[dimension.get()];
        for (int i = 0; i < dimension.get(); i++) {
            FloatWritable floatValue = new FloatWritable(dataInput.readFloat());
            features[i] = new FloatWritable(floatValue.get());
        }
    }

    //parsing of the features values on a String
    @Override
    public String toString(){
        String pointString = "";
        for (int i = 0; i < dimension.get(); i++){
            pointString += "," + features[i].get();
        }
        return pointString;
    }

    //computing the Euclidean distance
    public float computeDistance(Point centroid) {
        float sum = 0;
        for (int i = 0; i < dimension.get(); i++) {
            float difference = (centroid.features[i].get() - this.features[i].get());
            sum += (difference*difference);
        }
        return (float) Math.sqrt(sum);
    }

    //setting the value of each feature with the ones of the input array
    public void setFeatures(FloatWritable[] features_) {
        this.features = new FloatWritable[dimension.get()];

        for (int i = 0; i < dimension.get(); i++) {
            this.features[i] = new FloatWritable(features_[i].get());
        }
    }

    public float getKthFeature(int position) {
        return features[position].get();
    }

    public void setSumCounter(int counter) {
        this.sumCounter.set(counter);
    }

    public int getSumCounter() {
        return sumCounter.get();
    }

    //multiplication of each feature for the 'sumCounter' field
    public void computeMultiplicationOnFeatures() {
        for (int i = 0; i < dimension.get(); i++) {
            this.features[i].set(this.features[i].get()*sumCounter.get());
        }
    }

    public int getDimension() {
        return dimension.get();
    }

    public void setDimension(int dimension) {
        this.dimension.set(dimension);
    }
}


