import org.apache.spark.ml.feature.StringIndexer

object Helper {

  def index(columns: String*): Array[StringIndexer] = {
    columns.map(createIndexer).toArray
  }

  private def createIndexer(column: String): StringIndexer = {
    new StringIndexer().setInputCol(column).setOutputCol(s"${column}Index").setHandleInvalid("keep")
  }
}
