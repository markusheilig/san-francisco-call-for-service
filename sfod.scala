// dataset can be found at https://data.sfgov.org/Public-Safety/Fire-Department-Calls-for-Service/nuek-vuh3

// initialize spark session
import org.apache.spark.sql.SparkSession
import spark.implicits._

val spark = SparkSession.builder().getOrCreate()

// read dataset
val csvFile = "/Volumes/TranscendJetdriveLite330/downloads/Fire_Department_Calls_for_Service.csv"
var df = spark.read.option("header", "true").option("inferSchema", "true").csv(csvFile)

// data processing - rename columns (remove spaces and dashes)
val newColumnNames = df.columns.map(columnName => columnName.replace(" ", "").replace("-", ""))
df = df.toDF(newColumnNames: _*)

// data processing - remove all entries where label or feature columns are null
df = df.na.drop(Array("CallTypeGroup", "ZipcodeofIncident", "CallType", "NeighborhooodsAnalysisBoundaries", "NumberofAlarms", "EntryDtTm"))

// data processing - date schema is not detected automatically -> convert string to datetime
import org.apache.spark.sql.functions.to_timestamp
df = df.withColumn("EntryDtTm", to_timestamp($"EntryDtTm", "MM/dd/yyyy hh:mm:ss a"))

// data processing - get hour of incoming call
df = df.withColumn("EntryHour", hour($"EntryDtTm"))

// data processing - get quarter of incoming call
df = df.withColumn("EntryQuarter", quarter($"EntryDtTm"))

// data processing - drop columns where there is no time specification
df = df.na.drop(Array("EntryHour", "EntryQuarter"))

// data processing - add the label column (label = 1 <==> CallTypeGroup = "Potentially Life-Threatening" otherwise label = 0)
df = df.withColumn("label", when($"CallTypeGroup" === "Potentially Life-Threatening", 1).otherwise(0))

// data exploration - print schema
df.printSchema()

// data exploration - pretty print first row
{
  val colnames = df.columns
  val firstrow = df.head(1)(0)
  println("\r\nExample Data Row")
  for(ind <- Range(1, colnames.length)) {
    println(s"${colnames(ind)}: ${firstrow(ind)}\r\n")
  }
}

// data exploration - how many incidents are "Potentially Life-Threatening" (in percent)
{  
  val potentiallyLifeThreatening = df.filter($"label" === 1).count()  
  val percentage = potentiallyLifeThreatening.toDouble / df.count()
  print("%.2f%% of all incdidents are potentially life-threatening".format(percentage * 100))
}

// data processing - create String Indexer for categorical values
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
val callTypeGroupIndexer = new StringIndexer().setInputCol("CallTypeGroup").setOutputCol("CallTypeGroupIndex").setHandleInvalid("keep")
val callTypeIndexer = new StringIndexer().setInputCol("CallType").setOutputCol("CallTypeIndex").setHandleInvalid("keep")
val neighborhooodsAnalysisBoundariesIndexer = new StringIndexer().setInputCol("NeighborhooodsAnalysisBoundaries").setOutputCol("NeighborhooodsAnalysisBoundariesIndex").setHandleInvalid("keep")
val indexers = Array(callTypeGroupIndexer, callTypeIndexer, neighborhooodsAnalysisBoundariesIndexer)

// data processing - create One Hot Encoder for categorical values
val callTypeGroupEncoder = new OneHotEncoder().setInputCol("CallTypeGroupIndex").setOutputCol("CallTypeGroupVec")
val callTypeEncoder = new OneHotEncoder().setInputCol("CallTypeIndex").setOutputCol("CallTypeVec")
val neighborhooodsAnalysisBoundariesEncoder = new OneHotEncoder().setInputCol("NeighborhooodsAnalysisBoundariesIndex").setOutputCol("NeighborhooodsAnalysisBoundariesVec")
val encoders = Array(callTypeGroupEncoder, callTypeEncoder, neighborhooodsAnalysisBoundariesEncoder)

// (label, features)
import org.apache.spark.ml.feature.VectorAssembler
val features = Array("ZipcodeofIncident", "CallTypeVec", "NeighborhooodsAnalysisBoundariesVec", "EntryHour", "EntryQuarter", "NumberofAlarms")
val assembler = new VectorAssembler().setInputCols(features).setOutputCol("features")
val Array(training, test) = df.randomSplit(Array(0.8, 0.2), seed=12345)

// setup pipeline
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression

val lr = new LogisticRegression().setMaxIter(10)
val stages = indexers ++ encoders :+ assembler :+ lr
val pipeline = new Pipeline().setStages(stages)

// train
val model = pipeline.fit(training)

// test
val results = model.transform(test)

// model evaluation
import org.apache.spark.mllib.evaluation.MulticlassMetrics

val predictionAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd

val metrics = new MulticlassMetrics(predictionAndLabels)
println("Accuracy: ")
println(metrics.accuracy)
println("Confusion matrix: ")
println(metrics.confusionMatrix)
