import ch.hsr.geohash.GeoHash
import ch.hsr.geohash.GeoHash
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType

import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.SVMModel

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
    // dataset can be found at https://data.sfgov.org/Public-Safety/Fire-Incidents/wr8u-xric
    val fireCsv = "./../datasets/Fire_Incidents_100k.csv"

    var df = spark.read.option("header", "true").option("inferSchema", "true").csv(csvFile)
    var fire_incidents = spark.read.option("header", "true").option("inferSchema", "true").csv(fireCsv)

    //df = df.join(fire_incidents, df("Incident Number") === fire_incidents("Incident Number"))

    // data processing - rename columns (remove spaces and dashes)
    val newColumnNames = df.columns.map(_.replace(" ", "").replace("-", ""))
    df = df.toDF(newColumnNames: _*)

    // data processing - remove all entries where label or feature columns are null
    df = df.na.drop(Array("CallType", "Priority"))

    // data processing - date schema is not detected automatically -> convert string to datetime
    df = df.withColumn("EntryDtTm", to_timestamp($"EntryDtTm", "MM/dd/yyyy hh:mm:ss a"))
    df = df.withColumn("HourOfDay", hour($"EntryDtTm"))
    df = df.withColumn("DayOfYear", dayofyear($"EntryDtTm"))
    df = df.withColumn("isWeekend", dayofweek($"EntryDtTm") >= 5)
    df = df.withColumn("MonthOfYear", month($"EntryDtTm"))
    df = df.withColumn("WeekOfYear", weekofyear($"EntryDtTm"))

    // data processing - drop columns where there is no time specification
    df = df.na.drop(Array("HourOfDay", "isWeekend", "MonthOfYear", "WeekOfYear"))


    // Introduce geohash
    val pattern = "-?\\d+\\.{1}\\d+".r
    spark.udf.register("geohash", (s: String) => {
      var coords = new ListBuffer[String]
      pattern.findAllMatchIn(s).foreach(m => coords += m.toString())
      GeoHash.geoHashStringWithCharacterPrecision(coords.apply(0).toDouble, coords.apply(1).toDouble, 6)
    })
    df = df.withColumn("geohash", callUDF("geohash", $"Location"))
    df.show(20, false)

    /*{
      df.createOrReplaceTempView("incidents")
      println("Different isCriticalDispo : " + spark.sql("SELECT i1x.IncidentNumber, i2x.IncidentNumber FROM incidents as i1x INNER JOIN incidents as i2x ON i1x.IncidentNumber = i2x.IncidentNumber AND i1x.isCriticalDisposition != i2x.isCriticalDisposition ").count())
    }*/

    {
      def cdfUDF = udf((s:String) => {
        val criticalDispositions = Set("Code 2 Transport", "Fire", "Patient Declined Transport", "Against Medical Advice", "Medical Examiner")
        if (criticalDispositions.contains(s)) true else false
      } )

      def hospitalUDF = udf((hospital: String, transport: String) => {
        if (hospital == null && transport == null) false else true
      } )
      df = df.withColumn("isCriticalDispositionT", cdfUDF(df("CallFinalDisposition")))
        .withColumn("isHospitalTransportT", hospitalUDF(df("HospitalDtTm") , df("TransportDtTm")))

      var dfTemp = df.select($"IncidentNumber", $"isHospitalTransportT", $"isCriticalDispositionT")
      dfTemp = dfTemp.groupBy("IncidentNumber")
        .agg(collect_set('isCriticalDispositionT) as "tmpCD", collect_set('isHospitalTransportT) as "tmpH")
        .withColumn("isCriticalDisposition", array_contains('tmpCD, true))
        .withColumn("isHospitalTransport", array_contains('tmpH, true))
        .drop("tmpCD").drop("tmpH")
      df = df.join(dfTemp, Seq("IncidentNumber"))
    }

    /*{
      df.createOrReplaceTempView("incidents")
      println("Different isCriticalDispo : " + spark.sql("SELECT i1.IncidentNumber, i2.IncidentNumber FROM incidents as i1 INNER JOIN incidents as i2 ON i1.IncidentNumber = i2.IncidentNumber AND i1.isCriticalDisposition != i2.isCriticalDisposition ").count())
    }*/


    //df.show(20, false)

    // data processing - add the label column (label = 1 <==> CallTypeGroup = "Potentially Life-Threatening" otherwise label = 0)
    df = df.withColumn("label", when($"CallTypeGroup" === "Potentially Life-Threatening" && ($"isCriticalDisposition" === true || $"isHospitalTransport" === true), 1).otherwise(0))

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
    val (indexers, encoders) = Helper.indexAndEncode("CallType", "Priority", "NeighborhooodsAnalysisBoundaries")
    val featureNames = Array("HourOfDay", "DayOfYear" /*,"isWeekend" "WeekOfYear", "MonthOfYear"*/) ++ indexers.map(_.getOutputCol) //++ encoders.flatMap(_.getOutputCols)
    val assembler = new VectorAssembler().setInputCols(featureNames).setOutputCol("features")

    //Prepare Stages
    val stages = indexers ++ encoders :+ assembler

    //Print Features + Lable table
    val testPipe = new Pipeline().setStages(stages)
    val testModel = testPipe.fit(df)
    val testFrame = testModel.transform(df)
    var columns: Array[Column] = featureNames.map(testFrame(_))
    columns = columns :+ testFrame("label")
    testFrame.select(columns: _*).show(20, false)

    //Show pre statistics
    val numFeatures = featureNames.length
    println("numFeatures --> " + numFeatures)
    println("trainCount --> " + train.count())
    println("testCount --> " + test.count())

    // Classifiers
    def randomForest(): PipelineStage = {

      //Find feature with max count for maxBins
      var max: Long = 0
      var count: Long = 0
      featureNames.foreach(feature => {
        count = testFrame.agg(countDistinct(feature).as("count")).collectAsList().get(0).getAs[Long](0)
        if (max < count) {
          max = count
        }
      })

      new RandomForestClassifier()
        .setImpurity("gini")
        .setMaxDepth(20)
        .setNumTrees(30)
        .setFeatureSubsetStrategy("auto")
        .setSeed(5043)
        .setMaxBins(max.toInt /*+10*/)
    }

    def MLP(): PipelineStage = {
      val layers = Array[Int](numFeatures, 3, 2)
      return new MultilayerPerceptronClassifier()
        .setLayers(layers)
        .setBlockSize(128)
        .setSeed(1234L)
        .setMaxIter(100)
    }

    def LogisticRegression(): PipelineStage = {
      return new LogisticRegression()
    }

    def svm(): PipelineStage = {
      new LinearSVC().setMaxIter(10).setRegParam(0.1)
    }

    val classifier = randomForest()
    //Setup pipeline
    val pipeline = new Pipeline().setStages(stages :+ classifier)
    val model = pipeline.fit(train)

    {
      // save model to file
      //val timestamp = System.currentTimeMillis / 1000
      //model.save(s"models/myclassifier-$timestamp")
    }

    def printStatistics(what: String, result: DataFrame): Unit = {
      print(s"\n\n$what statistics:\n")
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
