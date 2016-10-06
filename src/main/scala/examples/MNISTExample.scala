package examples

import algebra.RichIndexedSeq
import neuralnetwork.Network
import utils.{MNISTImageLoader, MNISTLabelLoader}

class MNISTExample {
  private[this] val batchSize   = 10
  private[this] val batchCount  = 60000 / batchSize

  private[this] val epochs = 30
  private[this] val learningRate = 0.1
  private[this] val lambda = 5.0

  private[this] val l1 = new MNISTImageLoader("data/train-images-idx3-ubyte.gz", batchSize)
  private[this] val l2 = new MNISTLabelLoader("data/train-labels-idx1-ubyte.gz", batchSize)

  private[this] val network = new Network(List(784, 30, 10), learningRate, lambda, None)

  def run(): Unit = {
    train()
    trainingAccuracy()
    testAccuracy()
  }

  private def train(): Unit = {
    var e = 0

    while (e < epochs) {
      var i = 0
      val it1 = l1.images
      val it2 = l2.labels

      while (i < batchCount) {
        val tmp1 = it1.next
        val tmp2 = it2.next

        val x = tmp1.reshape(784, batchSize)
        val y = tmp2.reshape(10, batchSize)

        network.train(x, y, 60000)

        i += 1
      }

      e += 1
      println(s"Epoch $e done")
    }
  }

  private def trainingAccuracy(): Unit = {
    val it1 = l1.images
    val it2 = l2.labels

    var hit = 0
    for(i <- 1 to batchCount) {
      val ys = it2.next.grouped(10)

      it1.next.grouped(784).foreach { vs =>
        val res = network.compute(vs.reshape(784, 1)).values.zipWithIndex.maxBy(_._1)._2
        val y = ys.next.zipWithIndex.maxBy(_._1)._2
        if (res == y)
          hit += 1
      }
    }

    println(f"* training accuracy: ${(hit.toDouble / l1.count) * 100}%1.2f %%")
  }

  private def testAccuracy(): Unit = {
    val t1  = new MNISTImageLoader("data/t10k-images-idx3-ubyte.gz", 1)
    val t2  = new MNISTLabelLoader("data/t10k-labels-idx1-ubyte.gz", 1)

    val it1 = t1.images
    val it2 = t2.labels

    var hit = 0
    for(i <- 1 to t1.count) {
      val res = network.compute(it1.next.reshape(784, 1)).values.zipWithIndex.maxBy(_._1)._2
      val y = it2.next.zipWithIndex.maxBy(_._1)._2

      if (res == y)
        hit += 1
    }

    println(f"* test accuracy: ${(hit.toDouble / t1.count) * 100}%1.2f %%")
  }
}
