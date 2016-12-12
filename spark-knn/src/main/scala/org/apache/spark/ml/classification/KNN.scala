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

class KNN(override val uid: String) extends Predictor[Vector, KNN, KNNModel] {
  def this() = this(Identifiable.randomUID("knn"))

  override def copy(extra: ParamMap): KNN = defaultCopy(extra)

  var clusterPoints = muMap[Vector, ArrayBuffer[VectorWithNorm]]()

  var clusterRadius = muMap[Vector, Double]()

  var clusterCenters: Array[Vector] = null

  var k: Int = 1

  def setK(value: Int): this.type = {
    k = value
    this
  }

  override def train(dataset: DataFrame): KNNModel = {

    val instances = extractLabeledPoints(dataset).map {
      case LabeledPoint(label: Double, features: Vector) => (label, features)
    }

    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE

    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

    val labelSchema = dataset.schema("label")

    val computeNumClasses: () => Int = () => {
      val Row(maxLabelIndex: Double) = dataset.agg(max(col("label").cast(DoubleType))).head()
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
    new KNNModel(uid, k, numClasses, clusterCenters, clusterPointsRDD.asInstanceOf[RDD[(Vector, MAP[VectorWithNorm, Double])]], clusterRadiusRDD)
  }
}

class KNNModel(override val uid: String,
               val k: Int,
               val _numClasses: Int,
               val clusterCenters: Array[Vector],
               val clusterPointsRDD: RDD[(Vector, MAP[VectorWithNorm, Double])],
               val clusterRadiusRDD: RDD[(Vector, Double)])
                                                          extends ProbabilisticClassificationModel[Vector, KNNModel] {

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

        for(i <- 0 to Q.size-1) {
          var temp = Q.dequeue()
          if(predictMap.contains(temp.label)) {
            predictMap.put(temp.label, predictMap.get(temp.label).get + 1)
          }
          else {
            predictMap.put(temp.label, 1)
          }
        }

        var count: Int = Int.MinValue
        var predictedLabel:Double = 0.0
        for((key, value) <- predictMap) {
          if(count < value) {
            count = value
            predictedLabel = key
          }
        }

        var vector = new Array[Double](k)
        for(i <- 0 to Q.size-1) {
          vector(i) += Q.dequeue().label
        }

        val values = new ArrayBuffer[Any]

        val rawPrediction = Vectors.dense(vector)
        lazy val probability = raw2probability(rawPrediction)
        lazy val prediction = predictedLabel

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

  override def copy(extra: ParamMap): KNNModel = {
    val copied = new KNNModel(uid, k, numClasses, clusterCenters, clusterPointsRDD, clusterRadiusRDD)
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
        throw new RuntimeException("Unexpected error in KNNClassificationModel:" +
          " raw2probabilitiesInPlace encountered SparseVector")
    }
  }

  override protected def predictRaw(features: Vector): Vector = {
    throw new SparkException("predictRaw function should not be called directly since kNN prediction is done in distributed fashion. Use transform instead.")
  }
}

/**
  * VectorWithNorm can use more efficient algorithm to calculate distance
  */
case class VectorWithNorm(vector: Vector, norm: Double) {
  def this(vector: Vector) = this(vector, Vectors.norm(vector, 2))

  def this(vector: BV[Double]) = this(Vectors.fromBreeze(vector))

  def fastSquaredDistance(v: VectorWithNorm): Double = {
    KNNUtils.fastSquaredDistance(vector, norm, v.vector, v.norm)
  }

  def fastDistance(v: VectorWithNorm): Double = math.sqrt(fastSquaredDistance(v))
}

/**
  * VectorWithNorm plus auxiliary row information
  */
case class RowWithVector(vector: VectorWithNorm, row: Row) {
  def this(vector: Vector, row: Row) = this(new VectorWithNorm(vector), row)
  }
