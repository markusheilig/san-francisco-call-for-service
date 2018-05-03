package controllers

import org.apache.spark
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.month
import org.apache.spark.sql.functions.hour
import org.apache.spark.sql.functions.dayofweek
import org.apache.spark.sql.functions.weekofyear
import org.apache.spark.sql.functions.to_timestamp


class MachineModel() {

  implicit val spark: SparkSession = SparkSession.builder().config("spark.master", "local").getOrCreate()
  import spark.implicits._

  private val modelPath = "../core/models/myclassifier-1525355227"
  private val model = PipelineModel.load(modelPath)

  def isLifeThreatening(callType: String, priority: String, hood: String, dt: String): Boolean = {

    var df = spark.createDataFrame(Seq(
        (callType, priority, hood, dt)
      )).toDF("CallType", "Priority", "NeighborhooodsAnalysisBoundaries", "EntryDtTm")

    df = df.withColumn("EntryDtTm", to_timestamp($"EntryDtTm", "MM/dd/yyyy hh:mm:ss a"))
    df = df.withColumn("HourOfDay", hour($"EntryDtTm"))
    df = df.withColumn("DayOfWeek", dayofweek($"EntryDtTm"))
    df = df.withColumn("isWeekend", dayofweek($"EntryDtTm") >= 5)
    df = df.withColumn("MonthOfYear", month($"EntryDtTm"))
    df = df.withColumn("WeekOfYear", weekofyear($"EntryDtTm"))

    val res = model.transform(df)
    val result = res.select($"prediction").head()(0)
    val isLifeThreatening = result.asInstanceOf[Double].toInt == 1
    isLifeThreatening
  }

}
