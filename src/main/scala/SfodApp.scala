import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.{LogisticRegression, MultilayerPerceptronClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType

object SfodApp {

  def main(args: Array[String]): Unit = {
    // initialize spark session and run with all cores
    val cores = Runtime.getRuntime.availableProcessors()
    implicit val spark = SparkSession.builder().config("spark.master", s"local[$cores]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    import spark.implicits._

    // read dataset
    // dataset can be found at https://data.sfgov.org/Public-Safety/Fire-Department-Calls-for-Service/nuek-vuh3
    val csvFile = "./100k.csv"
    // dataset can be found at https://data.sfgov.org/Public-Safety/Fire-Incidents/wr8u-xric
    val fireCsv = "./Fire_Incidents.csv"
    /* Path
    val csvFile = "/Volumes/TranscendJetdriveLite330/downloads/Fire_Department_Calls_for_Service.csv"
     */
    var df = spark.read.option("header", "true").option("inferSchema", "true").csv(csvFile)
    var fire_incidents = spark.read.option("header", "true").option("inferSchema", "true").csv(fireCsv)

    df = df.join(fire_incidents, df("Incident Number") === fire_incidents("Incident Number"))

    // data processing - rename columns (remove spaces and dashes)
    val newColumnNames = df.columns.map(_.replace(" ", "").replace("-", ""))
    df = df.toDF(newColumnNames: _*)

    // data processing - remove all entries where label or feature columns are null
    df = df.na.drop(Array("CallType", "Priority", "ZipcodeofIncident"))

    // data processing - date schema is not detected automatically -> convert string to datetime
    df = df.withColumn("EntryDtTm", to_timestamp($"EntryDtTm", "MM/dd/yyyy hh:mm:ss a"))
    df = df.withColumn("HourOfDay", hour($"EntryDtTm"))
    df = df.withColumn("DayOfWeek", dayofweek($"EntryDtTm"))
    df = df.withColumn("isWeekend", dayofweek($"EntryDtTm") >= 5)
    df = df.withColumn("MonthOfYear", month($"EntryDtTm"))
    df = df.withColumn("WeekOfYear", weekofyear($"EntryDtTm"))

    // data processing - drop columns where there is no time specification
    df = df.na.drop(Array("HourOfDay", "isWeekend", "MonthOfYear", "WeekOfYear"))

    // data processing - add the label column (label = 1 <==> CallTypeGroup = "Potentially Life-Threatening" otherwise label = 0)
    df = df.withColumn("label", when($"CallTypeGroup" === "Potentially Life-Threatening", 1).otherwise(0))

    //There are no same incidentNumbers where CallTypeGroup or Priority is different (BUT slower and worse)
    //df = df.dropDuplicates("IncidentNumber")

    val Array(train, test) = df.randomSplit(Array(0.7, 0.3), seed = 12345)

    // data exploration - print schema
    df.printSchema()

    // data exploration - pretty print first row
    df.show(1, false)

    // data exploration - how many incidents are "Potentially Life-Threatening" (in percent)
    {
      val potentiallyLifeThreatening = df.filter($"label" === 1).count()
      val percentage = potentiallyLifeThreatening.toDouble / df.count()
      println("%.2f%% of all incidents are potentially life-threatening".format(percentage * 100))
    }

    // data processing - create String Indexer for categorical values
    val (indexers, encoders) = Helper.indexAndEncode("CallType", "Priority", "NeighborhooodsAnalysisBoundaries", "PrimarySituation")
    val featureNames = Array("HourOfDay", "DayOfWeek", "isWeekend", "WeekOfYear", "MonthOfYear", "ZipcodeofIncident") ++ indexers.map(_.getOutputCol) //++ encoders.flatMap(_.getOutputCols)
    val assembler = new VectorAssembler().setInputCols(featureNames).setOutputCol("features")

    //Prepare Stages
    val stages = indexers ++ encoders :+ assembler

    //Print Features + Lable table
    val testPipe = new Pipeline().setStages(stages)
    val testModel = testPipe.fit(df)
    val testFrame = testModel.transform(df)
    var columns: Array[Column] = featureNames.map(testFrame(_))
    columns = columns :+ testFrame("label")
    testFrame.select(columns:_*).show(20, false)

    //Show pre statistics
    val numFeatures = featureNames.length
    println("numFeatures --> " + numFeatures)
    println("trainCount --> " + train.count())
    println("testCount --> " + test.count())

    // Classifiers
    def randomForest(): PipelineStage ={

      //Find feature with max count for maxBins
      var max:Long = 0
      var count:Long = 0
      featureNames.foreach(feature => {
        count = testFrame.agg(countDistinct(feature).as("count")).collectAsList().get(0).getAs[Long](0)
        if (max < count) {
          max = count;
        }
      })

      return new RandomForestClassifier()
        .setImpurity("gini")
        .setMaxDepth(3)
        .setNumTrees(20)
        .setFeatureSubsetStrategy("auto")
        .setSeed(5043)
        .setMaxBins(max.toInt)
    }

    def MLP(): PipelineStage ={
      val layers = Array[Int](numFeatures, 7, 7 , 2)
      return new MultilayerPerceptronClassifier()
        .setLayers(layers)
        .setBlockSize(128)
        .setSeed(1234L)
        .setMaxIter(100)
    }

    def LogisticRegression(): PipelineStage ={
      return new LogisticRegression()
    }


    val classifier = randomForest()
    //Setup pipeline
    val pipeline = new Pipeline().setStages(stages :+ classifier)
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
