// dataset can be found at https://data.sfgov.org/Public-Safety/Fire-Department-Calls-for-Service/nuek-vuh3

// initialize spark session
import org.apache.spark.sql.SparkSession
import spark.implicits._

val spark = SparkSession.builder().getOrCreate()

// read dataset
val csvFile = "/Volumes/TranscendJetdriveLite330/downloads/Fire_Department_Calls_for_Service.csv"
var df = spark.read.option("header", "true").option("inferSchema", "true").csv(csvFile)

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

// data processing - rename columns (remove spaces)
val columnNamesWithoutSpaces = df.columns.map(columnName => columnName.replace(" ", ""))
df = df.toDF(columnNamesWithoutSpaces: _*)

// data processing - remove all entries where 'CallTypeGroup' is null
df = df.na.drop(Array("CallTypeGroup"))

// data processing - date schema is not detected automatically
import org.apache.spark.sql.functions.to_timestamp
df = df.withColumn("EntryDtTm", to_timestamp($"EntryDtTm", "MM/dd/yyyy hh:mm:ss a"))

// todo:
// EntryDtTm enthält komplettes datum, evtl. ist nur die aktuelle stunde wichtig (hh)
// zudem spielt evtl. der wochentag noch eine rolle? -> neue spalte erzeugen?

// data processing - add feature label (Feature = 1 <==> CallTypeGroup = "Potentially Life-Threatening" otherwise Feature = 0)
df = df.withColumn("Feature", when($"CallTypeGroup" === "Potentially Life-Threatening", 1).otherwise(0))

// data processing - create String Indexer for categorical values
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
df = new StringIndexer().setInputCol("CallTypeGroup").setOutputCol("CallTypeGroupIndex").fit(df).transform(df)
df = new StringIndexer().setInputCol("CallType").setOutputCol("CallTypeIndex").fit(df).transform(df)

// data processing - create One Hot Encoder for categorical values
df = new OneHotEncoder().setInputCol("CallTypeGroupIndex").setOutputCol("CallTypeGroupVec").transform(df)
df = new OneHotEncoder().setInputCol("CallTypeIndex").setOutputCol("CallTypeVec").transform(df)