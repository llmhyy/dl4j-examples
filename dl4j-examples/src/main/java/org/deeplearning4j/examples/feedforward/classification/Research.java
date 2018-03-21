package org.deeplearning4j.examples.feedforward.classification;

    import java.io.File;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * "Linear" Data Classification Example
 *
 * Based on the data from Jason Baldridge:
 * https://github.com/jasonbaldridge/try-tf/tree/master/simdata
 *
 * @author Josh Patterson
 * @author Alex Black (added plots)
 *
 */
public class Research {


    public static void main(String[] args) throws Exception {
        int seed = 123;
        double learningRate = 0.01;
        int batchSize = 50;
        int nEpochs = 100;

        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 20;

        final String filenameTrain  = new ClassPathResource("/data/train.csv").getFile().getPath();
        final String filenameTest  = new ClassPathResource("/data/test.csv").getFile().getPath();

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(filenameTrain)));
        MultiDataSetIterator trainIter = new RecordReaderMultiDataSetIterator.Builder(batchSize)
            .addReader("a",rr)
            .addInput("a",1,3)  //Input: columns 0 to 2 inclusive
            .addInput("a", 4,5)
            .addInput("a", 6,6)
            .addInput("a", 7, 109)
            .addInput("a", 110, 212)
            .addInput("a", 213, 315)
            .addInput("a", 316, 418)
            .addInput("a", 419, 521)
            .addInput("a", 522, 624)
            .addOutput("a",0,0) //Output: columns 3 to 4 inclusive
            .build();

        //Load the test/evaluation data:
        RecordReader rr0 = new CSVRecordReader();
        rr0.initialize(new FileSplit(new File(filenameTest)));
        MultiDataSetIterator testIter = new RecordReaderMultiDataSetIterator.Builder(batchSize)
            .addReader("myReader",rr0)
            .addInput("myReader",1,3)  //Input: columns 0 to 2 inclusive
            .addInput("myReader", 4,5)
            .addInput("myReader", 6,6)
            .addOutput("myReader",0,0) //Output: columns 3 to 4 inclusive
            .build();

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
            .learningRate(learningRate)
            .iterations(1)
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
			.updater(Updater.NESTEROVS)
            .graphBuilder()
            .addInputs("input1", "input2", "input3", "input4", "input5", "input6", "input7", "input8", "input9") //can use any label for this
            .addLayer("L1", new DenseLayer.Builder().nIn(3).nOut(1)
            		.weightInit(WeightInit.XAVIER)
					.activation(Activation.RELU).build(), "input1")
            .addLayer("L2", new DenseLayer.Builder().nIn(2).nOut(1)
            		.weightInit(WeightInit.XAVIER)
					.activation(Activation.RELU).build(), "input2")
            .addLayer("L3", new DenseLayer.Builder().nIn(1).nOut(1)
            		.weightInit(WeightInit.XAVIER)
					.activation(Activation.RELU).build(), "input3")
            .addLayer("L4", new DenseLayer.Builder().nIn(103).nOut(1)
            		.weightInit(WeightInit.XAVIER)
					.activation(Activation.RELU).build(), "input4")
            .addLayer("L5", new DenseLayer.Builder().nIn(103).nOut(1)
            		.weightInit(WeightInit.XAVIER)
					.activation(Activation.RELU).build(), "input5")
            .addLayer("L6", new DenseLayer.Builder().nIn(103).nOut(1)
            		.weightInit(WeightInit.XAVIER)
					.activation(Activation.RELU).build(), "input6")
            .addLayer("L7", new DenseLayer.Builder().nIn(103).nOut(1)
            		.weightInit(WeightInit.XAVIER)
					.activation(Activation.RELU).build(), "input7")
            .addLayer("L8", new DenseLayer.Builder().nIn(103).nOut(1)
            		.weightInit(WeightInit.XAVIER)
					.activation(Activation.RELU).build(), "input8")
            .addLayer("L9", new DenseLayer.Builder().nIn(103).nOut(1)
            		.weightInit(WeightInit.XAVIER)
					.activation(Activation.RELU).build(), "input9")
            .addVertex("merge", new MergeVertex(), "L1", "L2","L3", "L4", "L5","L6", "L7", "L8","L9")
            .addLayer("out",new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nIn(9).nOut(1)
            		.weightInit(WeightInit.XAVIER)
					.activation(Activation.SOFTMAX).build(), "merge")
            .setOutputs("out")	//We need to specify the network outputs and their order
            .pretrain(true)
            .backprop(true)
            .build();

        ComputationGraph model = new ComputationGraph(conf);
        model.init();
        
        System.out.println("before training");
        System.out.println(model.getLayer("L1").getParam("W"));
        System.out.println(model.getLayer("L1").getParam("b"));
        System.out.println(model.getLayer("out").getParam("W"));
        System.out.println(model.getLayer("out").getParam("b"));
        
        model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates

        for ( int n = 0; n < nEpochs; n++) {
            model.fit( trainIter );
        }

        System.out.println("after training");
        System.out.println(model.getLayer("L1").getParam("W"));
        System.out.println(model.getLayer("L1").getParam("b"));
        System.out.println(model.getLayer("out").getParam("W"));
        System.out.println(model.getLayer("out").getParam("b"));
        
        rr.initialize(new FileSplit(new File(filenameTrain)));
        trainIter = new RecordReaderMultiDataSetIterator.Builder(batchSize)
            .addReader("a",rr)
            .addInput("a",1,3)  //Input: columns 0 to 2 inclusive
            .addInput("a", 4,5)
            .addInput("a", 6,6)
            .addInput("a", 7, 109)
            .addInput("a", 110, 212)
            .addInput("a", 213, 315)
            .addInput("a", 316, 418)
            .addInput("a", 419, 521)
            .addInput("a", 522, 624)
            .addOutput("a",0,0) //Output: columns 3 to 4 inclusive
            .build();
        
        System.out.println("Evaluate model....");
        Evaluation trainEval = new Evaluation(numOutputs);
        while(trainIter.hasNext()){
            MultiDataSet t = trainIter.next();
            INDArray[] features = t.getFeatures();
            INDArray lables = t.getLabels(0);
            INDArray[] predicted = model.output(features);
//
            trainEval.eval(lables, predicted[0]);

        }

        //Print the evaluation statistics
        System.out.println(trainEval.stats());


        //------------------------------------------------------------------------------------
        //Training is complete. Code that follows is for plotting the data & predictions only

//        //Plot the data:
//        double xMin = -12.5;
//        double xMax = 12.5;
//        double yMin = -12.5;
//        double yMax = 12.5;
//
//        //Let's evaluate the predictions at every point in the x/y input space
//        int nPointsPerAxis = 100;
//        double[][] evalPoints = new double[nPointsPerAxis*nPointsPerAxis][2];
//        int count = 0;
//        for( int i=0; i<nPointsPerAxis; i++ ){
//            for( int j=0; j<nPointsPerAxis; j++ ){
//                double x = i * (xMax-xMin)/(nPointsPerAxis-1) + xMin;
//                double y = j * (yMax-yMin)/(nPointsPerAxis-1) + yMin;
//
//                evalPoints[count][0] = x;
//                evalPoints[count][1] = y;
//
//                count++;
//            }
//        }
//
//        INDArray allXYPoints = Nd4j.create(evalPoints);
//        INDArray predictionsAtXYPoints = model.output(allXYPoints);
//
//        //Get all of the training data in a single array, and plot it:
//        rr.initialize(new FileSplit(new ClassPathResource("/classification/train.csv").getFile()));
//        rr.reset();
//        int nTrainPoints = 1000;
//        trainIter = new RecordReaderDataSetIterator(rr,nTrainPoints,0,2);
//        DataSet ds = trainIter.next();
//        PlotUtil.plotTrainingData(ds.getFeatures(), ds.getLabels(), allXYPoints, predictionsAtXYPoints, nPointsPerAxis);
//
//
//        //Get test data, run the test data through the network to generate predictions, and plot those predictions:
//        rrTest.initialize(new FileSplit(new ClassPathResource("/classification/test.csv").getFile()));
//        rrTest.reset();
//        int nTestPoints = 500;
//        testIter = new RecordReaderDataSetIterator(rrTest,nTestPoints,0,2);
//        ds = testIter.next();
//        INDArray testPredicted = model.output(ds.getFeatures());
//        PlotUtil.plotTestData(ds.getFeatures(), ds.getLabels(), testPredicted, allXYPoints, predictionsAtXYPoints, nPointsPerAxis);
//
//        System.out.println("****************Example finished********************");
    }
}
