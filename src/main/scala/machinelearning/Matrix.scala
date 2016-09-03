package machinelearning

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

sealed trait MatrixLike[V] extends Serializable {
  def rows: Int
  def cols: Int
  def size = rows * cols

  def getValue(position: (Int, Int)): V

  def updated(position: (Int, Int), value: V): MatrixLike[V]
}

/**
 * Implementation of a MatrixLike using an IndexedSeq to store the data
 * internally.
 *
 * @tparam V a monoid
 */
//abstract class DenseMatrix[V: Monoid] extends MatrixLike[V] {
//  def rowsByColumns: IndexedSeq[V]
//
//  // This Seq is column-major, i.e. elements in the same column are next to each other in the array
//  // position = (column, row)
//  private[this] def tupToIndex(position: (Int, Int)): Int = position._1 * rows + position._2
//
//  override def getValue(position: (Int, Int)): V = rowsByColumns(tupToIndex(position))
//
//  override def updated(position: (Int, Int), value: V): MatrixLike[V] =
//    new Matrix[V](rows, cols, rowsByColumns.updated(tupToIndex(position), value))
//}


abstract class DenseMatrix[V: Monoid] extends MatrixLike[V] {
  def rowsByColumns: IndexedSeq[V]

  // This Seq is column-major, i.e. elements in the same column are next to each other in the array
  // position = (column, row)
  protected def tupToIndex(position: (Int, Int)): Int = position._1 * rows + position._2

  override def getValue(position: (Int, Int)): V = rowsByColumns(tupToIndex(position))

  override def updated(position: (Int, Int), value: V): DenseMatrix[V] =
    updated(rowsByColumns.updated(tupToIndex(position), value))

  def updated(values: IndexedSeq[V]): DenseMatrix[V]

  def map(f: V => V): Matrix[V] = new Matrix[V](rows, cols, rowsByColumns = rowsByColumns.map(f))

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
    new RowVector[V](rowValues(idx))
  }

  def rowValues(idx: Int): IndexedSeq[V] = {
    val res = IndexedSeq.newBuilder[V]
    var i = 0
    while (i < cols) {
      res += rowsByColumns(tupToIndex(i, idx))
      i += 1
    }
    res.result()
  }

  def swapRows(from: Int, to: Int): DenseMatrix[V] = {
    val res = rowsByColumns.toBuffer
    var i   = 0

    while(i < cols) {
      res(tupToIndex((i, to)))    = getValue((i, from))
      res(tupToIndex((i, from)))  = getValue((i, to))
      i += 1
    }

    updated(res.toIndexedSeq)
  }

  def mapRow(rowIdx: Int)(f: (V, Int) => V): DenseMatrix[V] = {
    val res = rowsByColumns.toBuffer
    var i   = 0

    while(i < cols) {
      val j = tupToIndex(i, rowIdx)
      res(j) = f(rowsByColumns(j), i)
      i += 1
    }

    updated(res.toIndexedSeq)
  }

  // Row operation
  def combineRows(rowIdx: Int, from: RowVector[V], modifier: V)(implicit field: Field[V]): DenseMatrix[V] = {
    mapRow(rowIdx)((v, idx) => field.plus(v, field.times(from(idx), modifier)))
  }

  def transpose: DenseMatrix[V] = {
    val res = IndexedSeq.newBuilder[V]

    var i = 0
    while(i < rows) {
      row(i).values.foreach(v => res += v)
      i += 1
    }

   updated(res.result())
  }
}

case class Matrix[V: Monoid](rows: Int, cols: Int, rowsByColumns: IndexedSeq[V]) extends DenseMatrix[V] {
  private[this] val valueMonoid = implicitly[Monoid[V]]

  /**
   * Apply the given function elements wise in both matrices
   */
  private[this] def elemWiseOp(that: Matrix[V])(fn: (V, V) => V): Matrix[V] = that match {
    case Matrix(r, c, elements) if r == rows && c == cols =>
      copy(rowsByColumns = (rowsByColumns, elements).zipped.map(fn))

    case _ => throw new IllegalArgumentException("matrix sizes does not match")
  }

  override def updated(values: IndexedSeq[V]): DenseMatrix[V] = Matrix(cols, rows, values)

  def +(that: Matrix[V]): Matrix[V] = elemWiseOp(that)(valueMonoid.plus)

  def -(that: Matrix[V])(implicit group: Group[V]) = elemWiseOp(that)(group.minus)

  def *[That, Res](that: That)(implicit product: MatrixProduct[Matrix[V], That, Res]): Res =
    product(this, that)

  def dropFirstColumn: Matrix[V] = {
    copy(cols = cols - 1, rowsByColumns = rowsByColumns.drop(rows))
  }

  def inverse(implicit f: Field[V], c: Ordering[V]): Matrix[V] = {
    if (rows == 2 && cols == 2)
      optimizedInverse2x2
    else
    {
      val identity = Matrix.identity[V](rows)
      val augmented = AugmentedMatrix(this, identity)

      MatrixOperation.gaussJordan(augmented).asInstanceOf[AugmentedMatrix[V]].rightMatrix
    }
  }

  private[this] def optimizedInverse2x2(implicit f: Field[V]): Matrix[V] = {
    val a = rowsByColumns.head
    val b = rowsByColumns(2)
    val c = rowsByColumns(1)
    val d = rowsByColumns(3)
    val numerator = f.minus(f.times(a, d), f.times(b, c))

    Matrix(2, 2, IndexedSeq(d, f.negate(c), f.negate(b), a)) * new Scalar[V](f.div(f.one, numerator))
  }
}

case class AugmentedMatrix[V: Field](
    leftRows: Int,
    leftCols: Int,
    rightRows: Int,
    rightCols: Int,
    rowsByColumns: mutable.IndexedSeq[V])
  extends DenseMatrix[V]
{
  override def updated(values: IndexedSeq[V]): DenseMatrix[V] = {
    copy(rowsByColumns = mutable.IndexedSeq(values: _*))
  }

  def leftMatrix: Matrix[V] =
    Matrix(leftRows, leftCols, rowsByColumns.take(leftRows * leftCols))

  def rightMatrix: Matrix[V] =
    Matrix(rightRows, rightCols, rowsByColumns.takeRight(rightCols * rightRows))

  override def rows: Int = leftRows
  override def cols: Int = leftCols + rightCols

  override def mapRow(rowIdx: Int)(f: (V, Int) => V): DenseMatrix[V] = {
    var i   = 0

    while(i < cols) {
      val j = tupToIndex(i, rowIdx)
      rowsByColumns(j) = f(rowsByColumns(j), i)
      i += 1
    }

    this
  }
}

object AugmentedMatrix {
  def apply[V](a: Matrix[V], b: Matrix[V])(implicit f: Field[V]): AugmentedMatrix[V] = {
    require(a.rows == b.rows)
    val cells = mutable.IndexedSeq(a.rowsByColumns ++ b.rowsByColumns: _*)
    new AugmentedMatrix[V](a.rows, a.cols, b.rows, b.cols, cells)
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

  def fill[V: Monoid](rows: Int, cols: Int)(f: => V): Matrix[V] = {
    new Matrix[V](rows, cols, IndexedSeq.fill(rows * cols)(f))
  }
}

class RowVector[V: Monoid](val values: IndexedSeq[V]) {
  @inline final def apply(idx: Int) = values.apply(idx)

  def *[That, Res](that: That)(implicit product: MatrixProduct[RowVector[V], That, Res]): Res =
    product(this, that)
}

object ColVector {
  def vectorOfOnes[V](size: Int)(implicit f: Ring[V]): ColVector[V] =
    new ColVector[V](IndexedSeq.fill(size)(f.one))
}

class ColVector[V: Monoid](val values: IndexedSeq[V]) {
  @inline final def apply(idx: Int) = values.apply(idx)

  def *[That, Res](that: That)(implicit product: MatrixProduct[ColVector[V], That, Res]): Res =
    product(this, that)

  def -(other: ColVector[V])(implicit f: Field[V]): ColVector[V] = {
    new ColVector[V]((values, other.values).zipped.map((v1, v2) => f.minus(v1, v2)))
  }

  // Element-wise multiplication
  def **(other: ColVector[V])(implicit f: Ring[V]): ColVector[V] = {
    new ColVector[V]((values, other.values).zipped.map((v1, v2) => f.times(v1, v2)))
  }

  final def dimension = values.size
}

class Scalar[V](val value: V) extends Serializable {
  def *[That, Res](that: That)(implicit product: MatrixProduct[Scalar[V], That, Res]) =
    product(this, that)
}


