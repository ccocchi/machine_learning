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

  def dot[That, Res](that: That)(implicit product: MatrixProduct[MatrixLike[V], That, Res]): Res =
    product(this, that)

  protected def elementWiseOp(other: MatrixLike[V])(fn: (V, V) => V): IndexedSeq[V] = {
    assert(rowSize == other.rowSize && colSize == other.colSize)
    (values, other.values).zipped.map(fn)
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

    copy(values = res.result())
  }
}

case class MVector[V](values: IndexedSeq[V]) extends MatrixLike[V] {
  type T = MVector[V]

  val rowSize = values.size
  val colSize = 1

  def dup(vs: IndexedSeq[V]): MVector[V] = copy(values = vs)
  def transpose: MatrixLike[V] = new Matrix(1, rowSize, values)
}
