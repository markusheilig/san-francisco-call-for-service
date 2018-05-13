package controllers

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.hour
import org.apache.spark.sql.functions.dayofweek
import org.apache.spark.sql.functions.to_timestamp


class MachineModel() {

  implicit val spark: SparkSession = SparkSession.builder().config("spark.master", "local").getOrCreate()
  import spark.implicits._

  private val modelPath = "../models/best"
  private val model = PipelineModel.load(modelPath)

  def isLifeThreatening(callType: String, priority: String, hood: String, dt: String): Boolean = {

    var df = spark.createDataFrame(Seq(
        (callType, priority, hood, dt)
      )).toDF("CallType", "FinalPriority", "NeighborhooodsAnalysisBoundaries", "EntryDtTm")

    df = df.withColumn("EntryDtTm", to_timestamp($"EntryDtTm", "MM/dd/yyyy hh:mm:ss a"))
    df = df.withColumn("HourOfDay", hour($"EntryDtTm"))
    df = df.withColumn("DayOfYear", dayofweek($"EntryDtTm"))

    val res = model.transform(df)
    val result = res.select($"prediction").head()(0)
    val isLifeThreatening = result.asInstanceOf[Double].toInt == 1
    isLifeThreatening
  }

}
