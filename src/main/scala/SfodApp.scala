import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.attribute.AttributeKeys

// dataset can be found at https://data.sfgov.org/Public-Safety/Fire-Department-Calls-for-Service/nuek-vuh3
object SfodApp {

  def main(args: Array[String]): Unit = {
    // initialize spark session and run with all cores
    val cores = Runtime.getRuntime.availableProcessors()
    val spark = SparkSession.builder().config("spark.master", s"local[$cores]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    import spark.implicits._

    // read dataset
    val csvFile = "./Fire_Department_Calls_for_Service_min.csv"
    /* Path
    val csvFile = "/Volumes/TranscendJetdriveLite330/downloads/Fire_Department_Calls_for_Service.csv"
     */
    var df = spark.read.option("header", "true").option("inferSchema", "true").csv(csvFile)

    // data processing - rename columns (remove spaces and dashes)
    val newColumnNames = df.columns.map(_.replace(" ", "").replace("-", ""))
    df = df.toDF(newColumnNames: _*)

    // data processing - remove all entries where label or feature columns are null
    df = df.na.drop(Array("CallTypeGroup", "ZipcodeofIncident", "CallType", "NeighborhooodsAnalysisBoundaries"))

    /*
    // data processing - date schema is not detected automatically -> convert string to datetime
    df = df.withColumn("EntryDtTm", to_timestamp($"EntryDtTm", "MM/dd/yyyy hh:mm:ss a"))

    // data processing - get time of day (0-6, 6-12, 12-18, 18-24)
    df = df.withColumn("TimeOfDay", hour($"EntryDtTm") / 6)

    // data processing - get quarter of incoming call
    df = df.withColumn("EntryQuarter", quarter($"EntryDtTm"))

    // data processing - drop columns where there is no time specification
    df = df.na.drop(Array("TimeOfDay", "EntryQuarter"))
    */

    // data processing - add the label column (label = 1 <==> CallTypeGroup = "Potentially Life-Threatening" otherwise label = 0)
    df = df.withColumn("label", when($"CallTypeGroup" === "Potentially Life-Threatening", 1).otherwise(0))

    val Array(train, test) = df.randomSplit(Array(0.7, 0.3), seed = 12345)

    // data exploration - print schema
    df.printSchema()

    // data exploration - pretty print first row
    {
      val colnames = df.columns
      val firstrow = df.head(1)(0)
      println("\r\nExample Data Row")
      for (ind <- Range(1, colnames.length)) {
        println(s"${colnames(ind)}: ${firstrow(ind)}\r\n")
      }
    }

    // data exploration - how many incidents are "Potentially Life-Threatening" (in percent)
    {
      val potentiallyLifeThreatening = df.filter($"label" === 1).count()
      val percentage = potentiallyLifeThreatening.toDouble / df.count()
      print("%.2f%% of all incidents are potentially life-threatening".format(percentage * 100))
    }

    // data processing - create String Indexer for categorical values
    val (indexers, encoders) = Helper.indexAndEncode("CallTypeGroup", "CallType", "NeighborhooodsAnalysisBoundaries")
    val featureNames = Array("ZipcodeofIncident") ++ indexers.map(_.getOutputCol) //++ encoders.flatMap(_.getOutputCols)
    val assembler = new VectorAssembler().setInputCols(featureNames).setOutputCol("features")

    // setup pipeline

    /*
    val lr = new LogisticRegression()
    val stages = indexers ++ encoders :+ assembler :+ lr
    val pipeline = new Pipeline().setStages(stages)

    // train
    val model = pipeline.fit(train)

    // test
    val results = model.transform(test)

    // model evaluation
    val predictionAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd

    val metrics = new MulticlassMetrics(predictionAndLabels)
    println("Accuracy: ")
    println(metrics.accuracy)
    println("Confusion matrix: ")
    println(metrics.confusionMatrix)
    */


    //val features = assembler.transform(df)
    val xxx = indexers ++ encoders :+ assembler
    val p = new Pipeline().setStages(xxx)
    val f = p.fit(df)
    f.transform(df).show(20, false)

    //val newTrainingData = train.select($"label", $"features")
    //val newTestData = test.select($"label", $"features")

    val numFeatures = featureNames.length
    println("numFeatures --> " + numFeatures)
    println("trainCount --> " + train.count())
    println("testCount --> " + test.count())
    val layers = Array[Int](numFeatures, 5,4, 2)
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)

    val stages = indexers :+ assembler :+ trainer
    val pipeline = new Pipeline().setStages(stages)

    // train the model
    val model = pipeline.fit(train)
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

    val metrics = new MulticlassMetrics(predictionAndLabels.as[(Double, Double)].rdd)
    println("Accuracy: ")
    println(metrics.accuracy)
    println("Confusion matrix: ")
    println(metrics.confusionMatrix)

  }
}
