package examples

import algebra.RichIndexedSeq
import algebra.MatrixLike
import neuralnetwork.Network
import scala.util.Random

class SpiralExample {
  private[this] val epochs = 1000
  private[this] val learningRate = 0.1
  private[this] val lambda = 0.7
  private[this] val dropout: Option[Double] = None

  private[this] val network = new Network(List(2, 100, 3), learningRate, lambda, dropout)

  def run(): Unit = {
    val (xs, ys) = generateData()

    val Y = Array.fill(ys.size * 3)(0.0)
    ys.zipWithIndex.foreach { case (z, i) =>  Y.update(i * 3 + z, 1.0) }

    val x = xs.reshape(2, 300)
    val y = Y.toIndexedSeq.reshape(3, 300)

    var e = 0

    while (e < epochs) {
      network.train(x, y, 300)
      e += 1
    }

    var hit = 0

    val it1 = xs.grouped(2)
    val it2 = ys.grouped(1)

    for(i <- 1 to 300) {
      val res = network.compute(it1.next().reshape(2, 1)).values.zipWithIndex.maxBy(_._1)._2
      val y = it2.next().head

      if (res == y)
        hit += 1
    }

    println(f"* training accuracy: ${(hit.toDouble / 300) * 100}%1.2f %%")
  }


  def generateData() = {
    val N = 100
    val D = 2
    val K = 3

    val X = IndexedSeq.newBuilder[Double]
    val y = IndexedSeq.fill(N)(0) ++ IndexedSeq.fill(N)(1) ++ IndexedSeq.fill(N)(2)

    for (j <- 0 until K) {
      val r = MatrixLike.linspace(0.0, 1, N)
      val t = MatrixLike.linspace(j * 4, (j + 1) * 4, N).map(n => n + Random.nextGaussian() * 0.2)

      val sin = (r.values, t.values).zipped.map((a, b) => a * Math.sin(b))
      val cos = (r.values, t.values).zipped.map((a, b) => a * Math.cos(b))

      (sin, cos).zipped.foreach((s, c) => X += (s, c))
    }

    (X.result(), y)
  }
}
