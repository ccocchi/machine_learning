package neuralnetwork

import machinelearning.{ColVector, Matrix, MatrixOperation}
import neuralnetwork.Network.Input

object Network {
  type Input = IndexedSeq[(IndexedSeq[Double], IndexedSeq[Double])]
}

class Network( val inputSize: Int,
               val outputSize: Int,
               val hiddenLayers: Int,
               val unitsPerLayer: Int,
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
    var deltas = layers.map(l => Matrix.fill(l.weightsMatrix.rows, l.weightsMatrix.cols)(0.0))
    var gradientBias = layers.map(l => new ColVector(IndexedSeq.fill(l.biasVector.dimension)(0.0)))

    input.foreach { case (x, y) =>
      val (ds, sigmas) = train(x, y)
      deltas = (deltas, ds).zipped.map(_ + _)
      gradientBias = (gradientBias, sigmas).zipped.map(_ + _)
    }

    val m = input.size
    val updatedWeights = (deltas, layers.map(_.weightsMatrix)).zipped.map { (dmat, wmat) =>
      val values = (dmat.rowsByColumns, wmat.rowsByColumns).zipped.map { (d, w) =>
        (1 - regularizationParameter * (learningRate / datasetSize)) * w - (learningRate / m) * d
      }
      Matrix(dmat.rows, dmat.cols, values)
    }

    (layers, updatedWeights, gradientBias).zipped.foreach { (l, mat, b) =>
       l.weightsMatrix = mat
       l.biasVector = l.biasVector - new ColVector(b.values.map(v => v * (learningRate / m)))
    }
  }

  /**
    * Train the network against a single element
    *
    * @param x The input values
    * @param y The output values
    * @return A matrix of deltas for each layer's weight matrix
    */
  def train(x: IndexedSeq[Double], y: IndexedSeq[Double]): (Seq[Matrix[Double]], Seq[ColVector[Double]]) = {
    val aValuesFull = activationValues(new ColVector(x)) // a1, a2, a3, a4

    val aValues = aValuesFull.tail.reverse // a4, a3, a2
    val thetas  = layers.map(_.weightsMatrix).reverse // theta3, theta2, theta1

    val a4 = aValues.head
    val initialError = aValues.head - new ColVector(y)

    val tailErrors = (aValues.tail, thetas).zipped.scanLeft(initialError) { case(err, (values, weights)) =>
      val merr = weights.transpose.asInstanceOf[Matrix[Double]] * err
      val vls  = values ** (ColVector.vectorOfOnes[Double](values.dimension) - values) // sp
      assert(merr.cols == 1)
      val errVec = new ColVector(merr.rowsByColumns)
      errVec ** vls
    }

    val errors = tailErrors.toIndexedSeq.reverse // sigma2, sigma3, sigma4

    val m = (aValuesFull.take(errors.size), errors).zipped.map { (a, s) =>
      val smat = Matrix(s.dimension, 1, s.values)
      val aTranspose = Matrix(1, a.dimension, a.values)
      smat * aTranspose
    }

    (m, errors)
  }

  def compute(inputs: ColVector[Double]): ColVector[Double] = {
    layers.foldLeft(inputs) { (values, l) => l.compute(values) }
  }

  def cost(input: Input): Double = {
    val m = input.size
    val partOne = input.map { case(inputs, expected) =>
      val result = compute(new ColVector(inputs))
      (result.values, expected).zipped.foldLeft(0.0) { case (res, (x, y)) => res + y * Math.log(x) + (1 - y) * Math.log(1 - x) }
    }

    val partTwo = layers.foldLeft(0.0) { (res, l) =>
      l.weightsMatrix.rowsByColumns.foldLeft(0.0)((acc, d) => acc + d * d)
    }

    (-partOne.sum / m) + (regularizationParameter * partTwo / (2 * m))
  }

  // List(a1, a2, a3, a4)
  private def activationValues(inputs: ColVector[Double]): Seq[ColVector[Double]] = {
    layers.scanLeft(inputs) { (values, l) => l.compute(values) }
  }

  private def buildLayers(): IndexedSeq[Layer] = {
    val seq = IndexedSeq.newBuilder[Layer]
    seq += new Layer(unitsPerLayer, inputSize)

    if (hiddenLayers > 1)
      seq ++= IndexedSeq.fill(hiddenLayers - 1) { new Layer(unitsPerLayer, unitsPerLayer) }

    seq += new Layer(outputSize, unitsPerLayer) // Output layer
    seq.result()
  }
}
