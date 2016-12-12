package org.apache.spark.ml.classification

import breeze.linalg.{Vector => BV}
import org.apache.spark.SparkException
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model, Predictor}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{ArrayType, DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._
import org.apache.spark.mllib.knn.KNNUtils
import org.apache.spark.ml.Predictor
import org.apache.spark.ml.util.MetadataUtils
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.annotation.Since
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.ml.clustering.KMeans

import scala.collection.mutable.{ArrayBuffer, LinkedHashMap, PriorityQueue}
import scala.collection.{Map => MAP}
import scala.collection.mutable.{Map => muMap}
import scala.{Double => scalaDouble}

class KNNRegression(override val uid: String) extends Predictor[Vector, KNNRegression, KNNRegressionModel] {
  def this() = this(Identifiable.randomUID("KNNRegression"))

  override def copy(extra: ParamMap): KNNRegression = defaultCopy(extra)

  var clusterPoints = muMap[Vector, ArrayBuffer[VectorWithNorm]]()

  var clusterRadius = muMap[Vector, Double]()

  var clusterCenters: Array[Vector] = null

  var centers: Array[Vector] = null

  var k: Int = 1

  def setK(value: Int): this.type = {
    k = value
    this
  }

  override def train(dataset: DataFrame): KNNRegressionModel = {

    val instances = extractLabeledPoints(dataset).map {
      case LabeledPoint(label: Double, features: Vector) => (label, features)
    }

    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE

    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

    val labelSchema = dataset.schema("label")

    val computeNumClasses: () => Int = () => {
      val Row(maxLabelIndex: Double) = dataset.agg(max(col("label").cast(DoubleType))).head()
      // classes are assumed to be numbered from 0,...,maxLabelIndex
      maxLabelIndex.toInt + 1
    }

    val numClasses = MetadataUtils.getNumClasses(labelSchema).fold(computeNumClasses())(identity)

    val points = instances.map{
      case (label, features) => (label, new VectorWithNorm(features))
    }

    var kVal = Math.min(50, Math.sqrt(dataset.count()/2).toInt)

    if(kVal>1)
      {
        val kMeans = new KMeans().setK(kVal)
        val kMeansModel = kMeans.fit(dataset)
        clusterCenters = kMeansModel.clusterCenters

        for(i <- 0 to clusterCenters.size-1) {
          clusterPoints.put(clusterCenters(i), new ArrayBuffer[VectorWithNorm]())
          clusterRadius.put(clusterCenters(i), Double.MaxValue)
        }
      }
    else {
      val kMeans = new KMeans().setK(2)
      val kMeansModel = kMeans.fit(dataset)
      clusterCenters = kMeansModel.clusterCenters
      clusterPoints.put(clusterCenters(0), new ArrayBuffer[VectorWithNorm]())
      clusterRadius.put(clusterCenters(0), Double.MaxValue)
    }

    // Map every point to thier respective cluster centers
    val intermediateRDD = points.map {
      case (label, point) =>
        var dist:Double = Double.MaxValue
        var curr_dist:Double = 0.0
        var curr_center:Vector = null
        for(value <- clusterCenters) {
          curr_dist = point.fastSquaredDistance(new VectorWithNorm(value))
          if(dist > curr_dist) {
            dist = curr_dist
            curr_center = value
          }
        }
        ((point, label), (curr_center, dist))
    }

    val clusterPointsRDD = intermediateRDD.map {
      case (pointWithLabel, (curr_center, dist)) =>
        (curr_center, pointWithLabel)
    }.groupByKey()

    val clusterRadiusRDD = intermediateRDD.map {
      case (point, value) =>
        value
    }.topByKey(1)
      .map{
        case (curr_center, distArray) =>
          (curr_center, distArray(0))
      }
    new KNNRegressionModel(uid, k, numClasses, clusterCenters, clusterPointsRDD.asInstanceOf[RDD[(Vector, MAP[VectorWithNorm, Double])]], clusterRadiusRDD)
  }
}

class KNNRegressionModel(override val uid: String,
               val k: Int,
               val _numClasses: Int,
               val clusterCenters: Array[Vector],
               val clusterPointsRDD: RDD[(Vector, MAP[VectorWithNorm, Double])],
               val clusterRadiusRDD: RDD[(Vector, Double)])
  extends ProbabilisticClassificationModel[Vector, KNNRegressionModel] {

  override def numClasses: Int = _numClasses

  override def transform(dataset: DataFrame): DataFrame = {

    val clusterPoints : MAP[Vector, MAP[VectorWithNorm, Double]] = clusterPointsRDD.collectAsMap()
    val clusterRadius : MAP[Vector, Double] = clusterRadiusRDD.collectAsMap()
    var inputPointsWithClusters = muMap[VectorWithNorm, ArrayBuffer[Vector]]()

    val features = dataset.select($(featuresCol))
      .map {
        r => new VectorWithNorm(r.getAs[Vector](0))
      }

    val merged = features.zipWithUniqueId().map {
      case (point, i) =>
        var dist:Double = Double.MaxValue
        var curr_dist:Double = 0.0
        var center:Vector = null
        var maxRadius:Double = 0.0

        for(value <- clusterCenters) {
          curr_dist = point.fastSquaredDistance(new VectorWithNorm(value))
          maxRadius = clusterRadius.get(value).get
          if(dist > (curr_dist + maxRadius)) {
            dist = curr_dist + maxRadius
            center = value
          }
        }
        val radius = clusterRadius.get(center).get
        var clustersMatched = ArrayBuffer[Vector]()

        inputPointsWithClusters.put(point, new ArrayBuffer[Vector]())

        if(curr_dist > clusterRadius.get(center).get)
          for((k, v) <- clusterRadius) {
            if(dist + radius >= point.fastSquaredDistance(new VectorWithNorm(k)) - v)
              clustersMatched += k
          }

        class HeapData(var dist: Double, var label: Double) extends Ordered[HeapData] {
          def compare(that: HeapData): Int = (this.dist) compare (that.dist)
        }

        val orderingHeap: Ordering[HeapData] = Ordering.by(e => e.dist)

        val Q = PriorityQueue[HeapData]()

        for (eachCenter <- clustersMatched) {
          for((eachClusterPoint, label) <- clusterPoints.get(eachCenter).get) {
            val dist = point.fastSquaredDistance(eachClusterPoint)
            var hd = new HeapData(dist , label)
            if(Q.size < k) {
              Q += hd
            }
            else if (dist < Q.head.dist) {
              Q.dequeue()
              Q += hd
            }
          }
        }

        var predictMap = muMap[Double, Int]()
        var sum : scalaDouble = 0.0
        var maxDist : scalaDouble = Int.MinValue
        var perfectMatchFound: Boolean = false
        var predictLabel : scalaDouble = 0.0

        for(I <- Q) {
          if(I.dist == 0) {
            perfectMatchFound = true
            predictLabel += I.label
          }
          if(!perfectMatchFound)
            sum += 1 / I.dist
          if(maxDist < I.dist)
            maxDist = I.dist
        }

        if(!perfectMatchFound) {
          for(I <- Q) {
            predictLabel += I.label / (I.dist * sum)
          }
        }

        lazy val prediction = predictLabel.toDouble

        val values = new ArrayBuffer[Any]

        if ($(predictionCol).nonEmpty) {
          values.append(prediction)
        }

        (i, values.toSeq)
    }

    dataset.sqlContext.createDataFrame(
      dataset.rdd.zipWithUniqueId().map { case (row, i) => (i, row) }
        .join(merged)
        .map {
          case (i, (row, values)) => Row.fromSeq(row.toSeq ++ values)
        },
      transformSchema(dataset.schema)
    )
  }

  override def transformSchema(schema: StructType): StructType = {
    var transformed = schema
    if ($(predictionCol).nonEmpty) {
      transformed = SchemaUtils.appendColumn(transformed, $(predictionCol), DoubleType)
    }
    transformed
  }

  override def copy(extra: ParamMap): KNNRegressionModel = {
    val copied = new KNNRegressionModel(uid, k, numClasses, clusterCenters, clusterPointsRDD, clusterRadiusRDD)
    copyValues(copied, extra).setParent(parent)
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        var i = 0
        val size = dv.size

        var sum = 0.0
        while (i < size) {
          sum += dv.values(i)
          i += 1
        }

        i = 0
        while (i < size) {
          dv.values(i) /= sum
          i += 1
        }

        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in KNNRegressionClassificationModel:" +
          " raw2probabilitiesInPlace encountered SparseVector")
    }
  }

  override protected def predictRaw(features: Vector): Vector = {
    throw new SparkException("predictRaw function should not be called directly since KNNRegression prediction is done in distributed fashion. Use transform instead.")
  }
}
