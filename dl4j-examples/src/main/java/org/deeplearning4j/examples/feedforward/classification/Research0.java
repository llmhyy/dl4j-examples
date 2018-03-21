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
import org.nd4j.linalg.dataset.api.MultiDataSet;
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
public class Research0 {


    public static void main(String[] args) throws Exception {
        int seed = 123;
        double learningRate = 0.01;
        int batchSize = 50;
        int nEpochs = 100;


        final String filenameTrain  = new ClassPathResource("/data/train.csv").getFile().getPath();
        final String filenameTest  = new ClassPathResource("/data/test.csv").getFile().getPath();

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(filenameTrain)));
        MultiDataSetIterator trainIter = new RecordReaderMultiDataSetIterator.Builder(batchSize)
            .addReader("myReader",rr)
            .addInput("myReader",1,3)  //Input: columns 0 to 2 inclusive
            .addOutput("myReader",0,0) //Output: columns 3 to 4 inclusive
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

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(1)
				.optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
				.learningRate(learningRate)
				.updater(Updater.NESTEROVS) // To configure: .updater(new Nesterovs(0.9))
				.list()
				.layer(0, new DenseLayer.Builder()
						.nIn(3)
						.nOut(1)
						.weightInit(WeightInit.XAVIER)
						.activation(Activation.SIGMOID)
						.build())
				.layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
						.weightInit(WeightInit.XAVIER)
						.activation(Activation.SOFTMAX)
						.weightInit(WeightInit.XAVIER)
						.nIn(1)
						.nOut(1)
						.build())
				.pretrain(false).backprop(true).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        
        System.out.println("before training");
        System.out.println(model.getLayer(0).getParam("W"));
        System.out.println(model.getLayer(0).getParam("b"));
        System.out.println(model.getLayer(1).getParam("W"));
        System.out.println(model.getLayer(1).getParam("b"));
        
        model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates

        for ( int n = 0; n < nEpochs; n++) {
            model.fit( trainIter );
        }

        System.out.println("after training");
        System.out.println(model.getLayer(0).getParam("W"));
        System.out.println(model.getLayer(0).getParam("b"));
        System.out.println(model.getLayer(1).getParam("W"));
        System.out.println(model.getLayer(1).getParam("b"));
        
        rr.initialize(new FileSplit(new File(filenameTrain)));
        trainIter = new RecordReaderMultiDataSetIterator.Builder(batchSize)
            .addReader("myReader",rr)
            .addInput("myReader",1,3)  //Input: columns 0 to 2 inclusive
            .addInput("myReader", 4,5)
            .addInput("myReader", 6,6)
            .addOutput("myReader",0,0) //Output: columns 3 to 4 inclusive
            .build();
        
        System.out.println("Evaluate model....");
        Evaluation trainEval = new Evaluation(2);
        while(trainIter.hasNext()){
            MultiDataSet t = trainIter.next();
            INDArray[] features = t.getFeatures();
            INDArray lables = t.getLabels(0);
            INDArray predicted = model.output(features[0]);
//
            trainEval.eval(lables, predicted);

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
