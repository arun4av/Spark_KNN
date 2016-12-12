import Common._

lazy val root = Project("spark-knn", file(".")).
  settings(commonSettings).
  aggregate(examples)

lazy val examples = knnProject("spark-knn-examples").
  settings(
    name := "spark-knn",
    spName := "saurfang/spark-knn",
    credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials"),
    licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")
  ).
  settings(Dependencies.examples)
