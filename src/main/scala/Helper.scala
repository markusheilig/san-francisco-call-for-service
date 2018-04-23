import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoderEstimator

object Helper {

  /**
   * Create StringIndexer and OneHotEncoderEstimators for the given columns
   * @param columns Category columns for which indexers and estimators will be created.
   * @return Tuple of indexers and estimators.
   */
  def indexAndEncode(columns: String*): (Array[StringIndexer], Array[OneHotEncoderEstimator]) = {
    val indexer = columns.map(createIndexer).toArray
    val encoders = columns.map(createEncoder).toArray
    (indexer, encoders)
  }

  private def createIndexer(column: String): StringIndexer = {
    new StringIndexer().setInputCol(column).setOutputCol(s"${column}Index").setHandleInvalid("keep")
  }

  private def createEncoder(column: String): OneHotEncoderEstimator = {
    new OneHotEncoderEstimator().setInputCols(Array(s"${column}Index")).setOutputCols(Array(s"${column}Vec"))
  }
}
