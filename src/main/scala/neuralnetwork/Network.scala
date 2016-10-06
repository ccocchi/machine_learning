package neuralnetwork

import scala.annotation.tailrec
import algebra.{MatrixLike, MatrixOperation}

class Network(layerConfiguration: List[Int],
              var learningRate: Double,
              regularizationParameter: Double,
              dropout: Option[Double] = None
             )
{
  private[this] lazy val layers = buildLayers()

  /**
    * Generic weight update function while doing the gradient descent
    * @param w Old weight value
    * @param d Delta calculated
    * @param n Size of the training set
    * @param r Learning rate scaled for the current batch
    */
  private def gradientDescentF(w: Double, d: Double, n: Int, r: Double) = (1 - (learningRate * regularizationParameter / n)) * w - (d * r)

  def train(x: MatrixLike[Double], y: MatrixLike[Double], n: Int): Unit = {
    val values = layers.scanLeft(x)((i, l) => l.compute(i)) // x, a2, a3

    val initialSigma = values.last - y // s3

    val sigmas = (layers.tail.reverse, values.reverse.tail).zipped.scanLeft(initialSigma) { case(sigma, (l, value)) =>
      l.sigma(value, sigma)
    } // s3, s2

    val errors = sigmas.toSeq.reverse
    val deltas = (values, errors).zipped.map((a, s) => s.dot(a.transpose))

    val batchLearningRate = learningRate / x.colSize
    val f = gradientDescentF(_: Double, _: Double, n, batchLearningRate)

    (layers, deltas, errors).zipped.foreach((l, delta, e) => l.update(delta, e, f, batchLearningRate))
  }

  def compute(x: MatrixLike[Double]): MatrixLike[Double] = layers.foldLeft(x)((i, l) => l.compute(i))

  def cost(x: MatrixLike[Double], y: MatrixLike[Double]): Double = {
    assert(x.colSize == y.colSize)

    val m = x.colSize
    val left = (compute(x).values, y.values).zipped.foldLeft(0.0) { case (res, (xx, yy)) =>
      res + (yy * Math.log(xx) + (1 - yy) * Math.log(1 - xx))
    }

    val right = layers.foldLeft(0.0) { (res, l) =>
      l.weightValues.foldLeft(0.0)((acc, d) => acc + d * d)
    }

    -left / m + (regularizationParameter * right / (2 * m))
  }

  private def buildLayers(): Seq[Layer] = {
    @tailrec
    def inner(l: List[Int], acc: List[Layer], inputSize: Int): List[Layer] = l match {
      case head :: Nil  => new Layer(head, inputSize) :: acc // output
      case head :: tail => // hidden
        dropout match {
              case Some(p) => inner(tail, new DropoutLayer(head, inputSize, p) :: acc, head)
          case None    => inner(tail, new Layer(head, inputSize) :: acc, head)
        }
      case Nil => acc
    }

    inner(layerConfiguration.tail, Nil, layerConfiguration.head).reverse
  }
}
