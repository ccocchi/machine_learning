package algebra

object MatrixOperation {
  def print[V](matrix: MatrixLike[V]): Unit = {
    val interpolation = List.fill(matrix.colSize)("%s").mkString("\t")

    for (i <- 0 until matrix.rowSize) {
      val res = Seq.newBuilder[V]

      for (j <- 0 until matrix.colSize) {
        res += matrix(i, j)
      }
      println(interpolation.format(res.result(): _*))
    }
  }
}
