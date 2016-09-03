package neuralnetwork

import machinelearning.{ColVector, Matrix, MatrixOperation}
import neuralnetwork.Network.Input

object Network {
  type Input = IndexedSeq[(IndexedSeq[Double], IndexedSeq[Double])]

  val regularizationParameter: Double = 0.01
  val learningRate = 0.1
}

class Network( val inputSize: Int,
               val outputSize: Int,
               val hiddenLayers: Int,
               val unitsPerLayer: Int
             )
{
  lazy val layers = buildLayers()

//  def train(input: Input): Unit = {
//    input.foreach { case (inputs, result) => train(inputs, result) }
//  }
//
//  def train(x: IndexedSeq[Double], y: IndexedSeq[Double]): Unit = {
//    val inputs = new ColVector(x)
//    val result = compute(inputs)
//
//    layers.foldRight[LayerWithResult]((None, Some(result))) { case (l1, l2) => l1.computeError(l2); (Some(l1), None) } // maybe l2 here but I don't think so
//    layers.foldLeft(inputs) { case (vs, l) => l.computeDelta(inputs); l.currentValue.get }
//  }


  /**
    * Train the network with given input
    *
    * @param input Dataset of inputs and results
    * @return The partial detivative value of the cost function
    */
  def train(input: Input): Double = {
    var deltas = layers.map(l => Matrix.fill(l.weightsMatrix.rows, l.weightsMatrix.cols)(0.0))

//    println("initial deltas")
//    deltas.foreach { d => MatrixOperation.print(d); println() }
//
//    println("weights")
//    layers.map(_.weightsMatrix).foreach { d => MatrixOperation.print(d); println() }

    input.foreach { case (x, y) =>
      val ds = train(x, y)
      deltas = (deltas, ds).zipped.map(_ + _)
//      { (acc, delt) =>
//        println("acc:")
//        MatrixOperation.print(acc)
//        println()
//
//        println("delta:")
//        MatrixOperation.print(delt)
//        println()
//        acc + delt
//      }
    }

//    println("trained deltas")
//    deltas.foreach { d => MatrixOperation.print(d); println() }

    val m = input.size
    val gradientMatrices = (deltas, layers).zipped.map { (d, l) =>
      val w = l.weightsMatrix

      val biasSeq = d.rowsByColumns.take(d.rows).map(v => v / m)
      val tailSeq = (d.rowsByColumns.drop(d.rows), w.rowsByColumns.drop(w.rows)).zipped.map((dv, wv) => (dv / m) + Network.regularizationParameter * wv )
      Matrix(d.rows, d.cols, biasSeq ++ tailSeq)
    }

    (layers, gradientMatrices).zipped.foreach { (l, m) =>
       l.weightsMatrix = l.weightsMatrix - Network.learningRate * m
    }

    gradientMatrices.map(m => m.rowsByColumns.map(Math.abs).sum).sum
  }

  /**
    * Train the network against a single element
    *
    * @param x The input values
    * @param y The output values
    * @return A matrix of deltas for each layer's weight matrix
    */
  def train(x: IndexedSeq[Double], y: IndexedSeq[Double]): Seq[Matrix[Double]] = {
    val aValuesFull = activationValues(new ColVector(x)) // a1, a2, a3, a4

//    println("activation values")
//    println(aValuesFull.map(_.values))

    val aValues = aValuesFull.tail.reverse // a4, a3, a2
    val thetas  = layers.map(_.weightsMatrix).reverse // theta3, theta2, theta1

    val initialError = aValues.head - new ColVector(y)
    println(s"initial error: ${initialError.values}")


//    val initialError = {
//      val values = (aValues.head.values, y).zipped.map((o, t) => (t - o) * (1 - o) * o)
//      new ColVector(values)
//    }

//    println("initial error")
//    println(initialError.values)

    val tailErrors = (aValues.tail, thetas).zipped.scanLeft(initialError) { case(err, (values, weights)) =>
      val merr = weights.dropFirstColumn.transpose.asInstanceOf[Matrix[Double]] * err
      val vls  = values ** (ColVector.vectorOfOnes[Double](values.dimension) - values)
      assert(merr.cols == 1)
      val errVec = new ColVector(merr.rowsByColumns)
      errVec ** vls
    }

    val errors = tailErrors.toIndexedSeq.reverse // sigma2, sigma3, sigma4

//    println("sigmas")
//    println(errors.map(_.values))

    (aValuesFull.take(errors.size), errors).zipped.map { (a, s) =>
      val smat = Matrix(s.dimension, 1, s.values)
      val aTranspose = Matrix(1, a.dimension + 1, 1.0 +: a.values)
      smat * aTranspose
    }
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

    (-partOne.sum / m) + (Network.regularizationParameter * partTwo / m)
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
