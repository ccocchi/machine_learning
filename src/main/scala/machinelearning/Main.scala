package machinelearning

object Main {
  def main (args: Array[String]) {
//    val matrix  = new Matrix(4, 3, IndexedSeq(1, 3, -3, 0, 2, 7, 1, 3, -3, 0, 2, 7))
//    val m       = new RowVector(IndexedSeq(1, 0, 1, 2, -1, 4))
//
//    var i = 0
//    while(i < 1000000) {
//      matrix.swapRows(3, 2)
//      i += 1
//    }
//
//
//    val start = System.currentTimeMillis()
//    i = 0
//    while(i < 1000000) {
//      matrix.swapRows(2, 3)
//      i += 1
//    }
//    val end = System.currentTimeMillis()
//    println(s"Time elaspsed: ${end - start}")

    val i = Matrix.identity[Double](3)

    val floatMatrix = new Matrix(3, 6, IndexedSeq(3.0, 1.0, 5.0, 2.0, -3.0, -1.0, -5.0, 2.0, 4.0) ++ i.rowsByColumns)
    MatrixOperation.print(floatMatrix)

    println()

    val m = MatrixOperation.gaussJordan(floatMatrix)
    MatrixOperation.print(m)
  }
}
