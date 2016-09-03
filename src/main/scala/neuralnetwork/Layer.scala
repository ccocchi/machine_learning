package neuralnetwork

import machinelearning.{ColVector, Matrix}
import neuralnetwork.Network.LayerWithResult

object Layer {
  final val epsilon = 0.0001
}

class Layer(size: Int, inputsSize: Int) {
  // List
  lazy val neurons = IndexedSeq.fill(size)(new Neuron)

//  var error: Option[ColVector[Double]] = None // sigma3

  /**
    * Values hold by this layer after computation.
    */
  var activationValues: Option[ColVector[Double]] = None // a3

  /**
    * Matrix used to compute activation values from input values. The +1 in columns size
    * is added for the bias input.
    * It is initialized with random values between -epsilon and +epsilon.
    */
  var weightsMatrix: Matrix[Double] = Matrix.fill(size, inputsSize + 1) {
    Math.random() * (2 * Layer.epsilon) - Layer.epsilon
  }

  //  var delta: Matrix[Double] = Matrix.zero(size, inputsSize) // delta2

  /**
    * Compute the values for the entire layer and store them in a var before returning them.
    *
    * @param inputs Values from the previous layer
    * @return A ColVector which size is this layer's size + 1
    */
  def compute(inputs: ColVector[Double]): ColVector[Double] = {
    val inputValues = 1.0 +: inputs.values
    val values = neurons.zipWithIndex.map { case (n, i) => n.compute(inputValues, weightsMatrix.rowValues(i)) }
    val vector = new ColVector(values)
    activationValues = Some(vector)
    vector
  }

//  def computeError(next: LayerWithResult): Unit = (next, error) match {
//    case ((Some(l), _), Some(v)) =>
//      val myValues = v ** (ColVector.vectorOfOnes[Double](size) - v)
//      val other    = l.theta.transpose.asInstanceOf[Matrix[Double]] * l.error.get
//      assert(other.cols == 1)
//      val vec = new ColVector(other.rowsByColumns)
//      error = Some(vec ** myValues)
//    case _ =>
//  }
//
//  def computeDelta(inputs: ColVector[Double]): Unit = error match {
//    case Some(e) =>
//      val errorMat  = Matrix(size, 1, e.values)
//      val inputsMat = Matrix(inputsSize, 1, inputs.values).transpose
//      delta = delta + errorMat * inputsMat
//
//    case _ =>
//  }
}

class OutputLayer(size: Int, inputsSize: Int) extends Layer(size, inputsSize) {
//  override def computeError(next: LayerWithResult): Unit = next match {
//    case (_, Some(res)) =>
//      error = Some(currentValue.get - res)
//    case _ =>
//  }
}
