package machinelearning

import scala.annotation.tailrec

object MatrixOperation {
//  def reduce[V: Field](m: Matrix[V]) = {
//    val valueField = implicitly[Field[V]]
//
//    @tailrec
//    def inner(matrix: Matrix[V], rowNumber: Int): Matrix[V] = rowNumber match {
//      case i if i < m.rows =>
//        val pivot = matrix.getValue((i, i))
//        val row   = matrix.row(i)
//
//        val m = (rowNumber + 1 to matrix.rows - 1).foldLeft(matrix) { (m, j) =>
//          val value = matrix.getValue((i, j))
//          m.combineRows(j, row, valueField.negate(valueField.div(value, pivot)))
//        }
//
//        inner(m, rowNumber + 1)
//
//      case _ => m
//    }
//
//    inner(m, 0)
//  }

  def print[V](matrix: DenseMatrix[V]): Unit = {
    val interpolation = List.fill(matrix.cols)("%s").mkString("\t")

    for (i <- 0 to matrix.rows - 1) {
      println(interpolation.format(matrix.row(i).values: _*))
    }
  }

  def gaussJordan[V](matrix: DenseMatrix[V])(implicit field: Field[V], compare: Ordering[V]): DenseMatrix[V] = {
    val limit = matrix match {
      case Matrix(_, c, _) => c
      case AugmentedMatrix(_, c, _, _, _) => c
    }

    @tailrec
    def reduce(m: DenseMatrix[V], row: Int, col: Int): DenseMatrix[V] = {
      if (col == limit) // m.cols, changed for testing
        return m

      val pivot  = m.getValue((row, col))
      val column = m.column(col).values

      pivot match {
        case p if p == field.zero && column.forall(v => v == field.zero) => reduce(m, row, col + 1)
        case p if p == field.zero =>
          val (_, idx) = column.zipWithIndex.max
          reduce(m.swapRows(row, idx), row, col)

        case p =>
          val newMatrix = m.mapRow(row)((v, _) => field.div(v, p))
          val nm = column.zipWithIndex.foldLeft(newMatrix) {
            case(mat, (_, idx)) if idx == row => mat
            case(mat, (v, idx)) =>
              val currentRow = newMatrix.row(row)
              mat.combineRows(idx, currentRow, field.negate(v))
          }
          reduce(nm, row + 1, col + 1)
      }
    }

    reduce(matrix, 0, 0)
  }
}
