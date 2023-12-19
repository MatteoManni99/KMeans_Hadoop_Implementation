package it.unipi.hadoop;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import static it.unipi.hadoop.Utils.*;

public class K_means
{
    public static class KMeansMapper extends Mapper<LongWritable, Text, IntWritable, Point> {
        //centroid array
        private Point[] centroids;
        //mapper output key
        private final IntWritable keyToReducer = new IntWritable();
        private int k;
        //array of centroid that remain void
        private final ArrayList<Integer> bannedK = new ArrayList<>();

        protected void setup(Context context) {
            Configuration conf = context.getConfiguration();
            //k setting
            k = conf.getInt("k", 0);
            centroids = new Point[k];

            //this is for manage the possibility of void centroids, to fill the bannedK from Config.
            String bannedKString = conf.get("bannedK");
            if(bannedKString != null){
                String[] bannedKStrings = bannedKString.split(",");
                for (String bannedKStringsElement : bannedKStrings) {
                    try {
                        int number = Integer.parseInt(bannedKStringsElement);
                        bannedK.add(number);
                    }catch (Exception e){
                        //nothing
                    }
                }
            }
            // retrieve centroids from previous iteration through config.
            for(int i=0; i<k; i++){
                if (bannedK.contains(i)) continue;
                centroids[i] = new Point(conf.get("centroid"+i));
            }
        }

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            //Initializing the output value
            Point valueToReducer = new Point(value.toString());
            //first iteration outside the 'for cycle' to compute the first distance value
            float min_distance = valueToReducer.computeDistance(centroids[0]);
            float new_distance;
            keyToReducer.set(0);

            //check the point distance from all other centroids
            for (int i = 1; i < k; i++) {
                //skip iteration if the 'i' corresponds to a centroid 'banned'
                if (bannedK.contains(i)) continue;
                new_distance = valueToReducer.computeDistance(centroids[i]);
                //update the key according to the nearest centroid
                if (new_distance < min_distance) {
                    min_distance = new_distance;
                    keyToReducer.set(i);
                }
            }
            context.write(keyToReducer, valueToReducer);
        }
    }

    //Reducer class
    public static class KMeansReducer extends Reducer<IntWritable, Point, IntWritable, Point> {

        protected void reduce(IntWritable key, Iterable<Point> values, Context context) throws IOException, InterruptedException {
            //parameter to keep track of the number of Points summed
            int tempSumCounter = 0;
            //Initializing output value
            Point temporaryCentroid = new Point();

            //Initializing the array on which we will perform the sum of the input list of Points
            FloatWritable[] tempFeatures = new FloatWritable[0];

            boolean firstIteration = true;

            for (Point point: values) {
                //with the check on this parameter we properly set the dimension and we initialize all its features to 0
                //this is done here because otherwise we wouldn't have a Point to get the data from
                if (firstIteration){
                    firstIteration = false;
                    int dimension = point.getDimension();
                    temporaryCentroid.setDimension(dimension);
                    temporaryCentroid.setSumCounter(0);

                    tempFeatures = new FloatWritable[dimension];
                    for (int i = 0; i < dimension; i++){
                        tempFeatures[i] = new FloatWritable(0);
                    }
                }

                int pointSumCounter = point.getSumCounter();
                //if this check is true, it means that the Point was already processed by the Combiner, so we have
                //to perform the multiplications on the Point's features by using its 'sumCounter' field
                if (pointSumCounter > 1) {
                    point.computeMultiplicationOnFeatures();
                }
                tempSumCounter += pointSumCounter;

                //performing the sum on each feature of the array we have created
                for (int i = 0; i < temporaryCentroid.getDimension(); i++){
                    tempFeatures[i].set(point.getKthFeature(i) + tempFeatures[i].get());
                }
            }

            //at this point we perform the division on each feature of the array we have created
            for (int i = 0; i < temporaryCentroid.getDimension(); i++){
                tempFeatures[i].set(tempFeatures[i].get()/tempSumCounter);
            }

            //setting the temporary centroid fields
            temporaryCentroid.setSumCounter(tempSumCounter);
            temporaryCentroid.setFeatures(tempFeatures);

            context.write(key, temporaryCentroid);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length != 5) {
            System.err.println("Usage: K_Means <input> <output> <k> <max_iteration> <threshold>");
            System.exit(1);
        }

        //parsing of the input parameters
        String inputPath = otherArgs[0];
        String outputPath = otherArgs[1];
        int k = Integer.parseInt(otherArgs[2]);
        int maxIteration = Integer.parseInt(otherArgs[3]);
        float threshold = Float.parseFloat(otherArgs[4]);

        System.out.println("args[0]: <input>="+inputPath);
        System.out.println("args[1]: <output>="+outputPath);
        System.out.println("args[2]: <k>="+k);
        System.out.println("args[3]: <maxIteration>="+maxIteration);
        System.out.println("args[4]: <threshold>="+threshold);

        //Initializing the array where we will keep track of the 'banned' centroids
        ArrayList<Integer> bannedK = new ArrayList<>();

        boolean stopCondition = false;
        int iteration = 0;
        boolean  thresholdCondition = false;

        long startTime = System.currentTimeMillis();

        //cycle for the iterations
        while(!stopCondition){
            //creating the output path where the computed centroids will be written
            String newCentroidsPath = outputPath + "/iteration" + iteration + "/";

            Job job = Job.getInstance(conf, "K_means");
            job.setJarByClass(K_means.class);

            job.setMapperClass(KMeansMapper.class);
            //setting the Combiner and Reducer class as the same one
            job.setCombinerClass(KMeansReducer.class);
            job.setReducerClass(KMeansReducer.class);

            job.setMapOutputKeyClass(IntWritable.class);
            job.setMapOutputValueClass(Point.class);

            job.setOutputKeyClass(IntWritable.class);
            job.setOutputValueClass(Point.class);

            //setting the parameters for the Config. file
            job.getConfiguration().setInt("k", k);
            String bannedKString = Arrays.toString(bannedK.toArray()).substring(1, Arrays.toString(bannedK.toArray()).length()-1);
            job.getConfiguration().set("bannedK", bannedKString);

            //setting the number of reducers
            job.setNumReduceTasks(k-bannedK.size());

            //define I/O
            FileInputFormat.addInputPath(job, new Path(inputPath));
            FileOutputFormat.setOutputPath(job, new Path(newCentroidsPath));

            job.setInputFormatClass(TextInputFormat.class);
            job.setOutputFormatClass(TextOutputFormat.class);

            if(iteration == 0){
                // In the first iteration set the centroids extracting k random point from the dataset
                List<String> initialCentroids = setInitialCentroid(FileSystem.get(job.getConfiguration()), inputPath, k);

                System.out.println("------------------------------------------------------------------------------------");
                for(int i=0; i<k; i++) {
                    //in the first iteration we set the centroids in the Config. file
                    job.getConfiguration().set("centroid" + i, initialCentroids.get(i));
                    System.out.println("initial centroid " + i + ": " + initialCentroids.get(i));
                }
                System.out.println("------------------------------------------------------------------------------------");

            }else{
                //path of the last computed centroids
                String oldCentroidsPath = outputPath + "/iteration" + (iteration-1) + "/";
                for(int i=0; i<k; i++){
                    //skipping 'for' iteration if the centroid corresponds to a 'banned' one
                    if (bannedK.contains(i)) continue;
                    //retrieving the features values of the centroid
                    String[] tokens = readOutputFile(FileSystem.get(job.getConfiguration()), oldCentroidsPath + "part-r-0000" + i);
                    String oldCentroidString = "";
                    for (int d = 1; d < tokens.length; d++){
                        oldCentroidString += tokens[d] + ",";
                    }
                    job.getConfiguration().set("centroid"+i, oldCentroidString.substring(0, oldCentroidString.length() - 1));
                }
            }

            job.waitForCompletion(true);

            //checking if the threshold is reached
            if (iteration > 0){
                thresholdCondition = checkCentroidsVariation(FileSystem.get(conf), outputPath, iteration, k, threshold, bannedK);
            }

            System.out.println("------------------------------------------------------------------------------------");
            System.out.println("thresholdCondition: " + thresholdCondition);
            System.out.println("iteration: " + iteration);
            System.out.println("bannedK: " + bannedK);
            System.out.println("------------------------------------------------------------------------------------");

            //checking if we have reached the stop condition
            if(iteration >= maxIteration || thresholdCondition){
                stopCondition = true;
            }

            iteration++;
        }

        //printing the time passed to perform the tests on the datasets
        long endTime = System.currentTimeMillis();
        long executionTime = endTime - startTime;
        System.out.println("Execution time: " + executionTime + " ms");
        System.out.println("------------------------------------------------------------------------------------");

    }
}
