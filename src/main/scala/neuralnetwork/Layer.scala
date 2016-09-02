package neuralnetwork

import machinelearning.{ColVector, Matrix}
import neuralnetwork.Network.LayerWithResult

object Layer {
  final val epsilon = 0.0001
}

/**
  * Created by ccocchi on 02/09/16.
  */
class Layer(size: Int, inputsSize: Int) {
  lazy val neurons = buildNeurons()

  var error: Option[ColVector[Double]] = None // sigma3
  var currentValue: Option[ColVector[Double]] = None // a3
  var theta: Matrix[Double] = Matrix(size, inputsSize, neurons.flatMap(_.weights)) // theta2
  var delta: Matrix[Double] = Matrix.zero(size, inputsSize) // delta2

  def compute(colVector: ColVector[Double]): ColVector[Double] = {
    val v = new ColVector(neurons.map(n => n.compute(colVector)))
    currentValue = Some(v)
    v
  }

  def computeError(next: LayerWithResult): Unit = (next, error) match {
    case ((Some(l), _), Some(v)) =>
      val myValues = v ** (ColVector.vectorOfOnes[Double](size) - v)
      val other    = l.theta.transpose.asInstanceOf[Matrix[Double]] * l.error.get
      assert(other.cols == 1)
      val vec = new ColVector(other.rowsByColumns)
      error = Some(vec ** myValues)
    case _ =>
  }

  def computeDelta(inputs: ColVector[Double]): Unit = error match {
    case Some(e) =>
      val errorMat  = Matrix(size, 1, e.values)
      val inputsMat = Matrix(inputsSize, 1, inputs.values).transpose
      delta = delta + errorMat * inputsMat

    case _ =>
  }

  protected def buildNeurons(): IndexedSeq[Neuron] = {
    IndexedSeq.fill(size) {
      val weights = IndexedSeq.fill(inputsSize) { Math.random() * (2 * Layer.epsilon) - Layer.epsilon }
      new Neuron(weights)
    }
  }
}

class OutputLayer(size: Int, inputsSize: Int) extends Layer(size, inputsSize) {
  override def computeError(next: LayerWithResult): Unit = next match {
    case (_, Some(res)) =>
      error = Some(currentValue.get - res)
    case _ =>
  }
}
