package controllers

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession

class MachineModel() {

  private val sparkSession = SparkSession.builder
    .master("local")
    .appName("ApplicationController")
    .getOrCreate()

  private val modelPath = "../core/models/myclassifier-1525355227"
  private val model = PipelineModel.load(modelPath)

  def isLifeThreatening(): Boolean = {
    true
  }

}
