package neuralnetwork

import machinelearning.ColVector

/**
  * Created by ccocchi on 02/09/16.
  */
class Neuron(var weights: IndexedSeq[Double]) {

  def inputsSize: Int = weights.size

  def compute(colVector: ColVector[Double]): Double = {
    val values = IndexedSeq(1.0) ++ colVector.values
    assert(values.size == weights.size, "Vector size not correct for neuron")
    val z = (values, weights).zipped.foldLeft(0.0) { case (acc, (v, w)) => acc + v * w }
    sigmoid(z)
  }

  final private def sigmoid(z: Double): Double = {
    1.0 / (1.0 - Math.exp(-z))
  }
}
