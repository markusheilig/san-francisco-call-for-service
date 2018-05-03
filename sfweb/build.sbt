
name := "sfweb"

version := "1.0"

scalaVersion := "2.11.8"

lazy val `sfweb` = (project in file(".")).enablePlugins(PlayScala)

resolvers += "scalaz-bintray" at "https://dl.bintray.com/scalaz/releases"

resolvers += "apache-snapshots" at "http://repository.apache.org/snapshots/"

resolvers += "Akka Snapshot Repository" at "http://repo.akka.io/snapshots/"


val sparkVersion = "2.3.0"

dependencyOverrides += "com.fasterxml.jackson.core" % "jackson-core" % "2.8.7"
dependencyOverrides += "com.fasterxml.jackson.core" % "jackson-databind" % "2.8.7"
dependencyOverrides += "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.8.7"

libraryDependencies ++= Seq( jdbc , ehcache , ws , guice,
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.hadoop" % "hadoop-client" % "2.7.2"
)
