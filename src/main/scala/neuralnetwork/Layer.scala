package neuralnetwork

import machinelearning.{ColVector, Matrix}

object Layer {
  final val epsilon = 0.001
}

class Layer(size: Int, inputsSize: Int) {
  // List
  lazy val neurons = IndexedSeq.fill(size)(new Neuron)

  /**
    * Matrix used to compute activation values from input values. The +1 in columns size
    * is added for the bias input.
    * It is initialized with random values between -epsilon and +epsilon.
    */
  var weightsMatrix: Matrix[Double] = Matrix.fill(size, inputsSize) {
    (Math.random() * (2 * Layer.epsilon) - Layer.epsilon) / Math.sqrt(inputsSize)
  }

  var biasVector = new ColVector(IndexedSeq.fill(size)(1.0))

  /**
    * Compute the values for the entire layer and store them in a var before returning them.
    *
    * @param inputs Values from the previous layer
    * @return A ColVector which size is this layer's size + 1
    */
  def compute(inputs: ColVector[Double]): ColVector[Double] = {
    val inputValues = inputs.values
    val biasValues = biasVector.values

    val values = neurons.zipWithIndex.map { case (n, i) =>
      n.compute(inputValues, weightsMatrix.rowValues(i), biasValues(i))
    }
    new ColVector(values)
  }
}
