package algebra

object MatrixOperation {
  def print[V](matrix: MatrixLike[V]): Unit = {
    for (i <- 0 until matrix.rowSize) {
      for (j <- 0 until matrix.colSize) {
        Console.printf("%2.2f\t", matrix(i, j))
      }
      println()
    }
  }
}
