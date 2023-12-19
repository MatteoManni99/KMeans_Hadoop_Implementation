package it.unipi.hadoop;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static java.lang.Math.abs;

public class Utils {

    //retrieving random lines from the input dataset to choose randomly the initial centroids
    public static List<String> setInitialCentroid(FileSystem fileSystem, String filePathString, int k) throws IOException {

        List<String> selectedLines = new ArrayList<>();
        Path filePath = new Path(filePathString);
        FSDataInputStream inputStream = fileSystem.open(filePath);

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
            List<String> allLines = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                allLines.add(line);
            }

            int totalLines = allLines.size();
            k = Math.min(k, totalLines);
            Random random = new Random();
            for (int i = 0; i < k; i++) {
                int randomIndex = random.nextInt(totalLines);
                selectedLines.add(allLines.get(randomIndex));
                allLines.remove(randomIndex);
                totalLines--;
            }
        }
        inputStream.close();
        fileSystem.close();
        return selectedLines;
    }

    //retrieving the feature values from a centroid file
    public static String[] readOutputFile(FileSystem fileSystem, String filePathString) throws IOException {
        Path filePath = new Path(filePathString);
        FSDataInputStream inputStream = fileSystem.open(filePath);
        String line;

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
            line = reader.readLine();
        }

        inputStream.close();
        return line.trim().split(",");
    }

    //computing the difference from centroids of consecutive iterations to check if the threshold is reached
    public static boolean checkCentroidsVariation(FileSystem fileSystem, String outputPath, int iteration, int k, float threshold, ArrayList<Integer> bannedK) throws IOException {
        String oldCentroidsPath = outputPath + "/iteration" + (iteration-1) + "/";
        String newCentroidsPath = outputPath + "/iteration" + (iteration) + "/";
        int dimension = 0;
        int count = 0;

        for(int i=0; i<k; i++){
            if (bannedK.contains(i)) continue;
            String[] oldValues;
            String[] newValues;

            oldValues = readOutputFile(fileSystem, oldCentroidsPath + "part-r-0000" + i);
            newValues = readOutputFile(fileSystem, newCentroidsPath + "part-r-0000" + i);

            for (int d = 1; d < oldValues.length; d++){
                dimension = oldValues.length-1;
                if (abs(Float.parseFloat(newValues[d]) - Float.parseFloat(oldValues[d])) < threshold)
                    count++;
            }
        }
        return count == dimension*(k-bannedK.size());
    }
}