package neuralnetwork

class Neuron {
  def compute(inputs: Seq[Double], weights: Seq[Double]): Double = {
    assert(inputs.size == weights.size, s"inputs have dimension ${inputs.size} and weights ${weights.size}")
    val z = (inputs, weights).zipped.foldLeft(0.0) { case (acc, (v, w)) => acc + (v * w) }
    //println(s"z: $z")
    val r = sigmoid(z)
    //println(s"r: $r")
    r
  }

  @inline
  final private def sigmoid(z: Double): Double = {
    1.0 / (1.0 + Math.exp(-z))
  }
}
