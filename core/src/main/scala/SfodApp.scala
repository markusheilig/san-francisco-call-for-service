import ch.hsr.geohash.GeoHash
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame, SparkSession}

import scala.collection.mutable.ListBuffer

object SfodApp {

  def main(args: Array[String]): Unit = {
    // initialize spark session and run with all cores
    val cores = Runtime.getRuntime.availableProcessors()
    implicit val spark = SparkSession.builder().config("spark.master", s"local[$cores]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    import spark.implicits._

    // read dataset
    // dataset can be found at https://data.sfgov.org/Public-Safety/Fire-Department-Calls-for-Service/nuek-vuh3
    val csvFile = "./../datasets/100k.csv"
    var df = spark.read.option("header", "true").option("inferSchema", "true").csv(csvFile)

    // data processing - rename columns (remove spaces and dashes)
    val newColumnNames = df.columns.map(_.replace(" ", "").replace("-", ""))
    df = df.toDF(newColumnNames: _*)

    // data processing - remove all entries where label or feature columns are null
    df = df.na.drop(Array("CallType", "Priority", "CallTypeGroup", "EntryDtTm"))

    // data processing - date schema is not detected automatically -> convert string to datetime
    df = df.withColumn("EntryDtTm", to_timestamp($"EntryDtTm", "MM/dd/yyyy hh:mm:ss a"))
    df = df.withColumn("HourOfDay", hour($"EntryDtTm"))
    df = df.withColumn("DayOfYear", dayofyear($"EntryDtTm"))
    df = df.na.drop(Array("HourOfDay", "DayOfYear"))

    // Introduce geohash
    val pattern = "-?\\d+\\.{1}\\d+".r
    spark.udf.register("geohash", (s: String) => {
      var coords = new ListBuffer[String]
      pattern.findAllMatchIn(s).foreach(m => coords += m.toString())
      GeoHash.geoHashStringWithCharacterPrecision(coords.apply(0).toDouble, coords.apply(1).toDouble, 6)
    })
    df = df.withColumn("geohash", callUDF("geohash", $"Location"))

    {
      // define our label

      def cdfUDF = udf((s:String) => {
        val criticalDispositions = Set("Code 2 Transport", "Fire", "Patient Declined Transport", "Against Medical Advice", "Medical Examiner", "Multi-casualty Incident")
        criticalDispositions.contains(s)
      } )

      def hospitalUDF = udf((hospital: String) => {
        if (hospital == null || hospital.isEmpty) false else true
      } )

      df = df.withColumn("isCriticalDispositionT", cdfUDF(df("CallFinalDisposition")))
        .withColumn("isHospitalTransportT", hospitalUDF(df("HospitalDtTm")))

      var dfTemp = df.select($"IncidentNumber", $"isHospitalTransportT", $"isCriticalDispositionT")
      dfTemp = dfTemp.groupBy("IncidentNumber")
        .agg(collect_set('isCriticalDispositionT) as "tmpCD", collect_set('isHospitalTransportT) as "tmpH")
        .withColumn("isCriticalDisposition", array_contains('tmpCD, true))
        .withColumn("isHospitalTransport", array_contains('tmpH, true))
        .drop("tmpCD").drop("tmpH")
      df = df.join(dfTemp, Seq("IncidentNumber"))

      // data processing - add the label column (label = 1 <==> CallTypeGroup = "Potentially Life-Threatening" otherwise label = 0)
      df = df.withColumn("label", when($"CallTypeGroup" === "Potentially Life-Threatening" && ($"isCriticalDisposition" === true || $"isHospitalTransport" === true), 1).otherwise(0))
    }

    val Array(train, test) = df.randomSplit(Array(0.9, 0.1), seed = 12345)

    // data processing - create String Indexer for categorical values
    val indexers = Helper.index("CallType", "Priority", "NeighborhooodsAnalysisBoundaries")
    val featureNames = Array("HourOfDay", "DayOfYear") ++ indexers.map(_.getOutputCol)
    val assembler = new VectorAssembler().setInputCols(featureNames).setOutputCol("features")

    //Prepare Stages
    val stages = indexers :+ assembler

    //Print Features + Lable table
    val testPipe = new Pipeline().setStages(stages)
    val testModel = testPipe.fit(df)
    val testFrame = testModel.transform(df)
    var columns: Array[Column] = featureNames.map(testFrame(_))
    columns = columns :+ testFrame("label")
    testFrame.select(columns: _*).show(20, false)
    // data exploration - how many incidents are "Potentially Life-Threatening" (in percent)
    {
      val potentiallyLifeThreatening = df.filter($"label" === 1).count()
      val percentage = potentiallyLifeThreatening.toDouble / df.count()
      println("%.2f%% of all incidents are potentially life-threatening".format(percentage * 100))
    }

    //Show pre statistics
    val numFeatures = featureNames.length
    println("numFeatures --> " + numFeatures)
    println("trainCount --> " + train.count())
    println("testCount --> " + test.count())

    // Classifiers
    def randomForest(): PipelineStage = {

      //Find feature with max count for maxBins
      var max: Long = 0
      featureNames.foreach(feature => {
        val count = testFrame.agg(countDistinct(feature).as("count")).collectAsList().get(0).getAs[Long](0)
        max = Math.max(max, count)
      })

      new RandomForestClassifier()
        .setImpurity("gini")
        .setMaxDepth(30)
        .setNumTrees(30)
        .setFeatureSubsetStrategy("auto")
        .setSeed(5043)
        .setMaxBins(max.toInt)
    }

    def multiLayerPerceptron(): PipelineStage = {
      val layers = Array[Int](numFeatures, 3, 2)
      new MultilayerPerceptronClassifier()
        .setLayers(layers)
        .setBlockSize(128)
        .setSeed(1234L)
        .setMaxIter(100)
    }

    def logisticRegression(): PipelineStage = {
      new LogisticRegression()
    }

    val classifier = randomForest()

    //Setup pipeline
    val pipeline = new Pipeline().setStages(stages :+ classifier)
    val model = pipeline.fit(train)

    // save model to file
    //val timestamp = System.currentTimeMillis / 1000
    //model.save(s"models/myclassifier-$timestamp")

    def printStatistics(resultType: String, result: DataFrame): Unit = {
      print(s"\n\n$resultType statistics:\n")
      val predictionAndLabels = result.select("prediction", "label")
      val metrics = new MulticlassMetrics(predictionAndLabels.as[(Double, Double)].rdd)
      val FPR = metrics.falsePositiveRate(0)
      val TPR = metrics.truePositiveRate(0)

      println(s"Accuracy: ${metrics.accuracy}")
      println(s"Confusion matrix: \n${metrics.confusionMatrix}")
      println(s"Precicion (How many selected items are relevant?): \n${metrics.precision(0)}")
      println(s"Recall (How many relevant items are selected?)): \n${metrics.recall(0)}")
      println(s"F-measure (best 1, worst 0): \n${metrics.fMeasure(0)}")
      println(s"Miss rate: \n${1 - TPR}")
      println(s"False alarm rate: \n${FPR}")

    }

    val trainResult = model.transform(train)
    printStatistics("Training", trainResult)

    val testResult = model.transform(test)
    printStatistics("Test", testResult)
  }
}
