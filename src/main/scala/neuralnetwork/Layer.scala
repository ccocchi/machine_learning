package neuralnetwork

import algebra.{MVector, MatrixLike}
import scala.util.Random

class Layer(size: Int, inputsSize: Int) {
  protected var weightsMatrix: MatrixLike[Double] = MatrixLike.fill(size, inputsSize) {
    Random.nextGaussian() / Math.sqrt(inputsSize)
  }

  protected var biasVector: MatrixLike[Double]    = MVector.fill(size)(0.0)

  /**
    * Compute the output of the layer
    *
    * @param inputs Input vector
    * @return A vector representing values for each node of the layer
    */
  def compute(inputs: MatrixLike[Double]): MatrixLike[Double] = {
    weightsMatrix.dot(inputs).plus(biasVector).map(activationFunction)
  }

  /**
    * Update the layer weight and bias. This method is called during backpropagation with
    * calculated deltas and sigmas, plus the gradient descent function applied to the
    * current batch.
    *
    * @param delta Matrix of delta for the weightsMatrix
    * @param e Vector og
    * @param f
    * @param c
    */
  def update(delta: MatrixLike[Double], e: MatrixLike[Double], f: (Double, Double) => Double, c: Double) = {
    weightsMatrix = weightsMatrix.mapWith(delta)(f)
    biasVector    = (biasVector - e.sumColumns * c).toMVector
  }

  /**
    * Calculate the sigmas for this layer and the current example(s).
    *
    * @param x Values computed by this layer for the current example(s)
    * @param previousSigma Sigma of the previous layer (in reverse order)
    * @return A matrix representing the sigma for this layer
    */
  def sigma(x: MatrixLike[Double], previousSigma: MatrixLike[Double]): MatrixLike[Double] = {
    val left  = weightsMatrix.transpose.dot(previousSigma)
    val ones  = MatrixLike.fill(x.rowSize, previousSigma.colSize)(1.0)
    val derivative = x * (ones - x)
    left * derivative
  }

  def weightValues = weightsMatrix.values

  private def activationFunction(z: Double): Double = 1.0 / (1.0 + Math.exp(-z))
}

class DropoutLayer(size: Int, inputsSize: Int, p: Double) extends Layer(size, inputsSize) {
  private[this] var mask = randomMask

  override def compute(inputs: MatrixLike[Double]): MatrixLike[Double] = {
    val result = super.compute(inputs)
    result.multiplicationPerColumn(mask)
  }

  override def update(delta: MatrixLike[Double], e: MatrixLike[Double], f: (Double, Double) => Double, c: Double): Unit = {
    var i, j = 0
    val builder = IndexedSeq.newBuilder[Double]

    while (j < weightsMatrix.colSize) {
      i = 0
      while (i < weightsMatrix.rowSize) {
        val w = weightsMatrix(i, j)

        if (mask(i, 0) == 0.0) {
          builder += w
        } else {
          val d = delta(i, j)
          builder += f(w, d)
        }

        i += 1
      }

      j += 1
    }

    weightsMatrix = MatrixLike(i, j, builder.result())
    biasVector    = (biasVector - e.sumColumns * mask * c).toMVector
    mask          = randomMask
  }

  private def randomMask = MVector.fill(size) {
    val r = Random.nextDouble()
    if (r < p)
      1.0
    else
      0.0
  }
}

class OutputLayer(size: Int, inputsSize: Int) extends Layer(size, inputsSize) {
}