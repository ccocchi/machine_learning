package utils

import java.io.{DataInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import scala.annotation.tailrec

class MNISTImageLoader(path: String) {
  private[this] var stream = new DataInputStream(new GZIPInputStream(new FileInputStream(path)))

  assert(stream.readInt() == 2051, "Wrong MNIST image stream magic")

  val count  = stream.readInt()
  val width  = stream.readInt()
  val height = stream.readInt()

  def images = imageStream(0)

  def rewind() = {
    stream.close()
    stream = new DataInputStream(new GZIPInputStream(new FileInputStream(path)))
    stream.skipBytes(16)
  }

  private def imageStream(i: Int): Stream[IndexedSeq[Double]] = {
    if (i >= count)
      Stream.empty
    else
      Stream.cons(readImage, imageStream(i + 1))
  }

  private def readImage: IndexedSeq[Double] = {
    val res = IndexedSeq.newBuilder[Double]

    for (i <- 0 until width)
      for (j <- 0 until height)
        res += stream.read().toDouble


    res.result()
  }
}

class MNISTLabelLoader(path: String) {
  private[this] var stream = new DataInputStream(new GZIPInputStream(new FileInputStream(path)))

  assert(stream.readInt() == 2049, "Wrong MNIST label stream magic")

  val count  = stream.readInt()

  def labels = labelStream(0)
  def rawLabels = rawLabelStream(0)

  def rewind() = {
    stream.close()
    stream = new DataInputStream(new GZIPInputStream(new FileInputStream(path)))
    stream.skipBytes(8)
  }

  private def rawLabelStream(i: Int): Stream[Int] = {
    if (i >= count)
      Stream.empty
    else
      Stream.cons(stream.readUnsignedByte(), rawLabelStream(i + 1))
  }

  private def labelStream(i: Int): Stream[IndexedSeq[Double]] = {
    if (i >= count)
      Stream.empty
    else
      Stream.cons(readLabel, labelStream(i + 1))
  }

  private def readLabel: IndexedSeq[Double] = {
    val res = Array.fill(10)(0.0)

    res.update(stream.readUnsignedByte(), 1.0)
    res.toIndexedSeq
  }
}



//class MNISTLoader(imageFilename: String, labelFilename: String) {
//  val iStream =
//  val lStream = new DataInputStream(new GZIPInputStream(new FileInputStream(labelFilename.toString)))




//  /**
//    * Load images from MNIST data set into a matrix. Each example is a column if the
//    * returned matrix.
//    *
//    * @return A matrix of dimension (feature number * dataset size)
//    */
//  def images: MatrixLike[Double] = {
//
//
//    assert(stream.readInt() == 2051, "Wrong MNIST image stream magic")
//

//
//    val res = IndexedSeq.newBuilder[Double]
//
//    for (n <- 0 until count)

//        }
//
//    res.result().reshape(width * height, count)
//  }

//  /**
//    * Load labels from MNIST data set into a matrix. Each vector contains a 1.0 in the jth
//    * position and 0 elsewhere, j corresponding to the corresponding label (0..9)
//    *
//    * @return A matrix of dimension (10 * dataset size)
//    */
//  def labels: MatrixLike[Double] = {
//
//
//    assert(stream.readInt() == 2049, "Wrong MNIST label stream magic")
//
//    val count = stream.readInt()
//    val res   = IndexedSeq.newBuilder[Double]
//
//    for(n <- 0 until count) {
//      val value = stream.readUnsignedByte()
//      val vector = Array.fill(10)(0.0)
//      vector.update(value, 1.0)
//      res ++= vector
//    }
//
//    res.result().reshape(10, count)
//  }
//}
//