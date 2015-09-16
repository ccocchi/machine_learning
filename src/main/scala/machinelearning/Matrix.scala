package machinelearning

abstract class MatrixLike[V: Monoid] extends Serializable {
  def rows: Int
  def cols: Int
  def size = rows * cols

  def getValue(position: (Int, Int)): V

  def updated(position: (Int, Int), value: V): MatrixLike[V]
}

case class Matrix[V: Monoid](rows: Int, cols: Int, rowsByColumns: IndexedSeq[V]) extends MatrixLike[V] {
  private[this] val valueMonoid = implicitly[Monoid[V]]

  private[this] def tupToIndex(position: (Int, Int)): Int = position._1 * cols + position._2

  override def getValue(position: (Int, Int)): V = rowsByColumns(tupToIndex(position))

  override def updated(position: (Int, Int), value: V): MatrixLike[V] =
    new Matrix[V](rows, cols, rowsByColumns.updated(tupToIndex(position), value))

  def +(other: Matrix[V]): Matrix[V] = other match {
    case Matrix(r, c, elements) if r == rows && c == cols =>
      new Matrix[V](r, c, (rowsByColumns, elements).zipped.map((x, y) => valueMonoid.plus(x, y)))

    case _ => throw new IllegalArgumentException("matrix sizes does not match")
  }
}


