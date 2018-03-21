package org.deeplearning4j.examples.feedforward.classification;

    import java.io.File;

    import org.datavec.api.records.reader.RecordReader;
    import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
    import org.datavec.api.split.FileSplit;
    import org.datavec.api.util.ClassPathResource;
    import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
    import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
    import org.deeplearning4j.eval.Evaluation;
    import org.deeplearning4j.nn.api.OptimizationAlgorithm;
    import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
    import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
    import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
    import org.deeplearning4j.nn.conf.Updater;
    import org.deeplearning4j.nn.conf.graph.MergeVertex;
    import org.deeplearning4j.nn.conf.layers.DenseLayer;
    import org.deeplearning4j.nn.conf.layers.GravesLSTM;
    import org.deeplearning4j.nn.conf.layers.OutputLayer;
    import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
    import org.deeplearning4j.nn.graph.ComputationGraph;
    import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
    import org.deeplearning4j.nn.weights.WeightInit;
    import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
    import org.deeplearning4j.util.ModelSerializer;
    import org.nd4j.linalg.activations.Activation;
    import org.nd4j.linalg.api.ndarray.INDArray;
    import org.nd4j.linalg.dataset.DataSet;
    import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
    import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
    import org.nd4j.linalg.factory.Nd4j;
    import org.nd4j.linalg.lossfunctions.LossFunctions;
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
        int nEpochs = 10;

        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 20;

        final String filenameTrain  = new ClassPathResource("/classification/control.csv").getFile().getPath();
        final String filenameTest  = new ClassPathResource("/classification/control2.csv").getFile().getPath();

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
//        rr.initialize(new FileSplit(new File("src/main/resources/classification/linear_data_train.csv")));
        rr.initialize(new FileSplit(new File(filenameTrain)));

        MultiDataSetIterator trainIter = new RecordReaderMultiDataSetIterator.Builder(batchSize)
            .addReader("myReader",rr)
            .addInput("myReader",1,2)  //Input: columns 0 to 2 inclusive
            .addOutput("myReader",0,0) //Output: columns 3 to 4 inclusive
            .build();

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(filenameTest)));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,2);

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input1") //can use any label for this
            .addLayer("L1",new DenseLayer.Builder().nIn(2).nOut(2).build(), "input1")
            .addLayer("L2",new DenseLayer.Builder().nIn(2).nOut(2).build(), "input1")
            .addLayer("L3",new DenseLayer.Builder().nIn(2).nOut(1).build(), "input1")
            .addLayer("L4",new DenseLayer.Builder().nIn(2).nOut(1).build(), "L1")
            .addLayer("L5",new DenseLayer.Builder().nIn(2).nOut(1).build(), "L2")
            .addLayer("L6",new DenseLayer.Builder().nIn(1).nOut(1).build(), "L3")
            .addVertex("merge", new MergeVertex(), "L4", "L5","L6")
            .addLayer("out",new OutputLayer.Builder().nIn(3).nOut(1).build(), "merge")
            .setOutputs("out")	//We need to specify the network outputs and their order
            .build();

        ComputationGraph model = new ComputationGraph(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates

        for ( int n = 0; n < nEpochs; n++) {
            model.fit( trainIter );
        }
        File modelFile = new File("E:\\model.zip");
//      ModelSerializer.restoreMultiLayerNetwork(modelFile);
        ModelSerializer.writeModel(model, modelFile, true);
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(modelFile);

        System.out.println("Evaluate model....");
        Evaluation testEval = new Evaluation(numOutputs);
        while(testIter.hasNext()){
            DataSet t = testIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
//            INDArray predicted = model.output(features,false);
//
//            testEval.eval(lables, predicted);

        }

        //Print the evaluation statistics
        System.out.println(testEval.stats());


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