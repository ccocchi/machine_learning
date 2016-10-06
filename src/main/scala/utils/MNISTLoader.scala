package utils

import java.io.{DataInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

class MNISTImageLoader(path: String, batchSize: Int) {
  private[this] val stream = new DataInputStream(new GZIPInputStream(new FileInputStream(path)))

  assert(stream.readInt() == 2051, "Wrong MNIST image stream magic")

  val count  = stream.readInt()
  val width  = stream.readInt()
  val height = stream.readInt()

  val head = imageStream(0)

  def images = head.iterator

  def imagesStream = imageStream(0)

  private def imageStream(i: Int): Stream[IndexedSeq[Double]] = {
    if (i >= count)
      Stream.empty
    else
      Stream.cons(readBatch, imageStream(i + 1))
  }

  private def readBatch: IndexedSeq[Double] = {
    val b = new Array[Byte](width * height * batchSize)
    stream.readFully(b)
    b.map(v => (v & 0xFF) / 255.0).toIndexedSeq
  }
}

class MNISTLabelLoader(path: String, batchSize: Int) {
  private[this] val stream = new DataInputStream(new GZIPInputStream(new FileInputStream(path)))

  assert(stream.readInt() == 2049, "Wrong MNIST label stream magic")

  val count  = stream.readInt()

  val head = labelStream(0)

  def labels = head.iterator

  private def labelStream(i: Int): Stream[IndexedSeq[Double]] = {
    if (i >= count)
      Stream.empty
    else
      Stream.cons(readLabel, labelStream(i + 1))
  }

  private def readLabel: IndexedSeq[Double] = {
    val res = Array.fill(10 * batchSize)(0.0)
    var i = 0
    while (i < batchSize) {
      res.update(stream.readUnsignedByte() + 10 * i, 1.0)
      i += 1
    }

    res.toIndexedSeq
  }
}