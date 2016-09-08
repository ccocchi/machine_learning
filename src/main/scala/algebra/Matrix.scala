package algebra

import algebra.maths.{Monoid, Group, Ring}

sealed trait MatrixLike[V] {
  val rowSize: Int
  val colSize: Int

  def values: IndexedSeq[V]
  def apply(row: Int, col: Int): V = values(col * rowSize + row)

  def dup(vs: IndexedSeq[V]): MatrixLike[V]

  def transpose: MatrixLike[V]

  def +(other: MatrixLike[V])(implicit m: Monoid[V]): MatrixLike[V] = dup(elementWiseOp(other)(m.plus))
  def -(other: MatrixLike[V])(implicit g: Group[V]): MatrixLike[V]  = dup(elementWiseOp(other)(g.minus))
  def *(other: MatrixLike[V])(implicit r: Ring[V]): MatrixLike[V]   = dup(elementWiseOp(other)(r.times))
  def *(scalar: V)(implicit r: Ring[V]): MatrixLike[V] = map(v => r.times(v, scalar))

  def dot[That, Res](that: That)(implicit product: MatrixProduct[MatrixLike[V], That, Res]): Res =
    product(this, that)

  /**
    * Add a vector like matrix to each column of the matrix.
    *
    * @param other
    * @param m
    * @return
    */
  def plus(other: MatrixLike[V])(implicit m: Monoid[V]): MatrixLike[V] = {
    assert(rowSize == other.rowSize && other.colSize == 1)

    val it = Iterator.continually(other.values).flatten
    val nw = values.map(v => m.plus(v, it.next()))
    dup(nw)
  }

  def minus(other: MatrixLike[V])(implicit g: Group[V]): MatrixLike[V] = {
    val it = Iterator.continually(other.values).flatten
    val nw = values.map(v => g.minus(v, it.next()))
    dup(nw)
  }

  def map(f: (V) => V): MatrixLike[V] = dup(values.map(f))
  def mapWith(other: MatrixLike[V])(f: (V, V) => V): MatrixLike[V] = dup(elementWiseOp(other)(f))

  def sumColumns(implicit m: Monoid[V]): MVector[V] = {
    val res = IndexedSeq.newBuilder[V]

    var i = 0
    while(i < rowSize) {
      var j = 0
      var tmp = m.zero

      while(j < colSize) {
        tmp = m.plus(tmp, this(i, j))
        j += 1
      }

      res += tmp
      i += 1
    }

    MVector(res.result())
  }

  protected def elementWiseOp(other: MatrixLike[V])(fn: (V, V) => V): IndexedSeq[V] = {
    assert(rowSize == other.rowSize && colSize == other.colSize, s"$dimensionsString != ${other.dimensionsString}")
    (values, other.values).zipped.map(fn)
  }

  def toMVector: MVector[V]

  def dimensionsString: String = s"${rowSize}x$colSize"

  def groupByColumns(n: Int): Seq[MatrixLike[V]] = {
    values.grouped(n * rowSize).map { vs => vs.reshape(rowSize, vs.size / rowSize) }.toSeq
  }

  def columnsValues: Seq[IndexedSeq[V]] = {
    values.grouped(rowSize).toSeq
  }
}

object MatrixLike {
  def fill[V](rows: Int, cols: Int)(f: => V): MatrixLike[V] = apply(rows, cols, IndexedSeq.fill(rows * cols)(f))

  def apply[V](rows: Int, cols: Int, values: IndexedSeq[V]): MatrixLike[V] = cols match {
    case 1 => MVector(values)
    case _ => Matrix(rows, cols, values)
  }
}

case class Matrix[V](rowSize: Int, colSize: Int, values: IndexedSeq[V]) extends MatrixLike[V] {
  type T = Matrix[V]

  def dup(vs: IndexedSeq[V]): Matrix[V] = copy(values = vs)

  def transpose: MatrixLike[V] = {
    val res = IndexedSeq.newBuilder[V]
    var i = 0

    while (i < rowSize) {
      var j = 0

      while (j < colSize) {
        res += this(i, j)
        j += 1
      }

      i += 1
    }

    Matrix(colSize, rowSize, res.result())
  }

  def toMVector: MVector[V] = {
    assert(colSize == 1)
    MVector(values)
  }
}

object MVector {
  def fill[V](n: Int)(f: => V): MVector[V] = MVector(IndexedSeq.fill(n)(f))
}

case class MVector[V](values: IndexedSeq[V]) extends MatrixLike[V] {
  type T = MVector[V]

  val rowSize = values.size
  val colSize = 1

  def dup(vs: IndexedSeq[V]): MVector[V] = copy(values = vs)
  def transpose: MatrixLike[V] = new Matrix(1, rowSize, values)

  def toMVector: MVector[V] = this
}
