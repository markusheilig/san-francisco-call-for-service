import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

// dataset can be found at https://data.sfgov.org/Public-Safety/Fire-Department-Calls-for-Service/nuek-vuh3
object SfodApp {

  def main(args: Array[String]): Unit = {
    // initialize spark session and run with all cores
    val cores = Runtime.getRuntime.availableProcessors()
    implicit val spark = SparkSession.builder().config("spark.master", s"local[$cores]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    import spark.implicits._

    // read dataset
    val csvFile = "./100k.csv"
    /* Path
    val csvFile = "/Volumes/TranscendJetdriveLite330/downloads/Fire_Department_Calls_for_Service.csv"
     */
    var df = spark.read.option("header", "true").option("inferSchema", "true").csv(csvFile)

    // data processing - rename columns (remove spaces and dashes)
    val newColumnNames = df.columns.map(_.replace(" ", "").replace("-", ""))
    df = df.toDF(newColumnNames: _*)

    // data processing - remove all entries where label or feature columns are null
    df = df.na.drop(Array("CallType","Priority", "ZipcodeofIncident"))
    
    // data processing - date schema is not detected automatically -> convert string to datetime
    df = df.withColumn("EntryDtTm", to_timestamp($"EntryDtTm", "MM/dd/yyyy hh:mm:ss a"))

    df = df.withColumn("HourOfDay", hour($"EntryDtTm"))
    df = df.withColumn("DayOfWeek", dayofweek($"EntryDtTm"))
    df = df.withColumn("isWeekend", dayofweek($"EntryDtTm")>=5)
    df = df.withColumn("MonthOfYear", month($"EntryDtTm"))
    df = df.withColumn("WeekOfYear", weekofyear($"EntryDtTm"))

    // data processing - drop columns where there is no time specification
    df = df.na.drop(Array("HourOfDay", "isWeekend", "MonthOfYear", "WeekOfYear"))

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
    val (indexers, encoders) = Helper.indexAndEncode("CallType", "Box", "Priority", "ZipcodeofIncident")
    val featureNames = Array("DayOfWeek","MonthOfYear","isWeekend", "HourOfDay", "WeekOfYear") ++ indexers.map(_.getOutputCol) //++ encoders.flatMap(_.getOutputCols)
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
    val layers = Array[Int](numFeatures, 7, 7 , 2)
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)

    val stages = indexers :+ assembler :+ trainer
    val pipeline = new Pipeline().setStages(stages)

    // train the model
    val model = pipeline.fit(train)

    def printStatistics(what: String, result: DataFrame): Unit = {
      print(s"\n\n$what statistics:\n")
      val predictionAndLabels = result.select("prediction", "label")
      val metrics = new MulticlassMetrics(predictionAndLabels.as[(Double, Double)].rdd)
      println(s"Accuracy: ${metrics.accuracy}")
      println(s"Confusion matrix: \n${metrics.confusionMatrix}")
    }

    val trainResult = model.transform(train)
    printStatistics("Training", trainResult)

    val testResult = model.transform(test)
    printStatistics("Test", testResult)

  }

}
