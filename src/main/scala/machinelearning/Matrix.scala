package machinelearning

import scala.collection.mutable.ArrayBuffer

abstract class MatrixLike[V: Monoid] extends Serializable {
  def rows: Int
  def cols: Int
  def size = rows * cols

  def getValue(position: (Int, Int)): V

  def updated(position: (Int, Int), value: V): MatrixLike[V]
}

case class Matrix[V: Monoid](rows: Int, cols: Int, rowsByColumns: IndexedSeq[V]) extends MatrixLike[V] {
  private[this] val valueMonoid = implicitly[Monoid[V]]

  // This Seq is column-major, i.e. elements in the same column are next to each other in the array
  private[this] def tupToIndex(position: (Int, Int)): Int = position._1 * rows + position._2

  override def getValue(position: (Int, Int)): V = rowsByColumns(tupToIndex(position))

  override def updated(position: (Int, Int), value: V): MatrixLike[V] =
    new Matrix[V](rows, cols, rowsByColumns.updated(tupToIndex(position), value))

  def map(f: V => V): Matrix[V] = copy(rowsByColumns = rowsByColumns.map(f))

  /**
   * Apply the given function elements wise in both matrices
   */
  private[this] def elemWiseOp(that: Matrix[V])(fn: (V, V) => V): Matrix[V] = that match {
    case Matrix(r, c, elements) if r == rows && c == cols =>
      copy(rowsByColumns = (rowsByColumns, elements).zipped.map(fn))

    case _ => throw new IllegalArgumentException("matrix sizes does not match")
  }

  def column(idx: Int): ColVector[V] = {
    val res = IndexedSeq.newBuilder[V]
    var i = 0
    while(i < rows) {
      res += rowsByColumns(tupToIndex(idx, i))
      i += 1
    }
    new ColVector[V](res.result())
  }

  def row(idx: Int): RowVector[V] = {
    val res = IndexedSeq.newBuilder[V]
    var i = 0
    while (i < cols) {
      res += rowsByColumns(tupToIndex(i, idx))
      i += 1
    }
    new RowVector[V](res.result())
  }

  def +(that: Matrix[V]): Matrix[V] = elemWiseOp(that)(valueMonoid.plus)

  def *[That, Res](that: That)(implicit product: MatrixProduct[Matrix[V], That, Res]): Res =
    product(this, that)

  def transpose: Matrix[V] = {
    val res = IndexedSeq.newBuilder[V]

    var i = 0
    while(i < rows) {
      row(i).values.foreach(v => res += v)
      i += 1
    }

    Matrix(cols, rows, res.result())
  }
}

object Matrix {
  def identity[V](i: Int)(implicit ring: Ring[V]): Matrix[V] = {
    val values = ArrayBuffer.fill(i * i)(ring.zero)
    var col = 0

    while (col < i) {
      val idx = col * i + col
      values.update(idx, ring.one)
      col += 1
    }

    Matrix(i, i, values.toIndexedSeq)
  }
}

class RowVector[V: Monoid](val values: IndexedSeq[V]) {
  def *[That, Res](that: That)(implicit product: MatrixProduct[RowVector[V], That, Res]): Res =
    product(this, that)
}

class ColVector[V: Monoid](val values: IndexedSeq[V]) {
  def *[That, Res](that: That)(implicit product: MatrixProduct[ColVector[V], That, Res]): Res =
    product(this, that)
}

class Scalar[V](val value: V) extends Serializable {
  def *[That, Res](that: That)(implicit product: MatrixProduct[Scalar[V], That, Res]) =
    product(this, that)
}


