package neuralnetwork

import machinelearning.{ColVector, Matrix}

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
}
