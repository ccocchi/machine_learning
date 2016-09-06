package neuralnetwork

import neuralnetwork.Network.Input
import algebra.{MVector, MatrixLike, RichIndexedSeq}

import scala.annotation.tailrec

object Network {
  type Input = IndexedSeq[(IndexedSeq[Double], IndexedSeq[Double])]
}

class Network( layerConfiguration: List[Int],
               var learningRate: Double,
               regularizationParameter: Double
             )
{
  lazy val layers = buildLayers()

  /**
    * Train the network with given input
    *
    * @param input Dataset of inputs and results
    * @return The partial detivative value of the cost function
    */
  def train(input: Input, datasetSize: Int): Unit = {
    var deltas = layers.map(l => MatrixLike.fill(l.weightsMatrix.rowSize, l.weightsMatrix.colSize)(0.0))
    var gradientBias = layers.map(l => MatrixLike.fill(l.biasVector.rowSize, 1)(0.0))

    input.foreach { case (x, y) =>
      val (ds, sigmas) = train(x, y)
      deltas = (deltas, ds).zipped.map(_ + _)
      gradientBias = (gradientBias, sigmas).zipped.map(_ + _)
    }

    val m = input.size
    val updatedWeights = (deltas, layers.map(_.weightsMatrix)).zipped.map { (dmat, wmat) =>
      dmat.mapWith(wmat) { (d, w) =>
        (1 - regularizationParameter * (learningRate / datasetSize)) * w - (learningRate / m) * d
      }
    }

    (layers, updatedWeights, gradientBias).zipped.foreach { (l, mat, b) =>
       l.weightsMatrix = mat
       l.biasVector = l.biasVector - MVector(b.values.map(v => v * (learningRate / m)))
    }
  }

  /**
    * Train the network against a single element
    *
    * @param x The input values
    * @param y The output values
    * @return A matrix of deltas for each layer's weight matrix
    */
  def train(x: IndexedSeq[Double], y: IndexedSeq[Double]): (Seq[MatrixLike[Double]], Seq[MatrixLike[Double]]) = {
    val xMat = x.reshape(x.size, 1)
    val aValuesFull = layers.scanLeft(xMat)((i, l) => l.compute(i)) // a1, a2, a3, a4

    val aValues = aValuesFull.tail.reverse // a4, a3, a2
    val thetas  = layers.map(_.weightsMatrix).reverse // theta3, theta2, theta1

    val a4 = aValues.head
    val initialError = aValues.head - y.reshape(y.size, 1)

    val tailErrors = (aValues.tail, thetas).zipped.scanLeft(initialError) { case(err, (values, weights)) =>
      val merr = weights.transpose.dot(err)
      val vls  = values * (MatrixLike.fill(values.rowSize, 1)(1.0) - values) // sp
      assert(merr.colSize == 1)
      merr * vls
    }

    val errors = tailErrors.toIndexedSeq.reverse // sigma2, sigma3, sigma4

    val m = (aValuesFull.take(errors.size), errors).zipped.map { (a, s) =>
      s.dot(a.transpose)
    }

    (m, errors)
  }

  def train(x: MatrixLike[Double], y: MatrixLike[Double], n: Int): Unit = {
    val values = layers.scanLeft(x)((i, l) => l.compute(i)) // x, a2, a3
    val initialSigma = values.last - y // s3

    val sigmas = (layers.tail.reverse, values.reverse.tail).zipped.scanLeft(initialSigma) { case(sigma, (l, value)) =>
      l.backprop(value, sigma)
    } // s3, s2

    val errors = sigmas.toSeq.reverse
    val deltas = (values, errors).zipped.map((a, s) => s.dot(a.transpose))

    (layers, deltas, errors).zipped.foreach { (l, delta, e) =>
      l.weightsMatrix = l.weightsMatrix.mapWith(delta) { (w, d) =>
        (1 - (learningRate * regularizationParameter / n)) * w - (d * learningRate / x.colSize)
      }

      l.biasVector = (l.biasVector - e.sumColumns.dot(learningRate / x.colSize)).toMVector
    }
  }

  def compute(x: MatrixLike[Double]): MatrixLike[Double] = layers.foldLeft(x)((i, l) => l.compute(i))

  def cost(x: MatrixLike[Double], y: MatrixLike[Double]): Double = {
    assert(x.colSize == y.colSize)

    val m = x.colSize
    val left = (compute(x).values, y.values).zipped.foldLeft(0.0) { case (res, (xx, yy)) =>
      res + (yy * Math.log(xx) + (1 - yy) * Math.log(1 - xx))
    }

    val right = layers.foldLeft(0.0) { (res, l) =>
      l.weightsMatrix.values.foldLeft(0.0)((acc, d) => acc + d * d)
    }

    -left / m + (regularizationParameter * right / (2 * m))
  }

  private def buildLayers(): Seq[Layer] = {
    @tailrec
    def inner(l: List[Int], acc: List[Layer], inputSize: Int): List[Layer] = l match {
      // case head :: Nil  => new OutputLayer(head, inputSize) :: acc
      case head :: tail => inner(tail, new Layer(head, inputSize) :: acc, head)
      case Nil => acc
    }

    inner(layerConfiguration.tail, Nil, layerConfiguration.head).reverse
  }
}
