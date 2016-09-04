package neuralnetwork

class Neuron {
  def compute(inputs: Seq[Double], weights: Seq[Double], bias: Double): Double = {
    assert(inputs.size == weights.size, s"inputs have dimension ${inputs.size} and weights ${weights.size}")
    val z = (inputs, weights).zipped.foldLeft(0.0) { case (acc, (v, w)) => acc + (v * w) } + bias
    sigmoid(z)
  }

  @inline
  final private def sigmoid(z: Double): Double = {
    1.0 / (1.0 + Math.exp(-z))
  }
}
