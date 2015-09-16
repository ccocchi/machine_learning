package machinelearning

import org.scalatest.WordSpec

class MatrixTest extends WordSpec {
  /*
    [ 1   0
      3   2
      -3  7 ]
   */
  val matrix = new Matrix(3, 2, IndexedSeq(1, 0, 3, 2, -3, 7))

  /*
    [ 2  -1
      0   4
      1   1 ]
   */
  val matrix2 = new Matrix(3, 2, IndexedSeq(2, -1, 0, 4, 1, 1))


  "A matrix" should {
    "have a given size" in {
      assert(matrix.size == 6)
    }

    "have elements accessible via position" in {
      assert(matrix.getValue((0, 0)) == 1)
      assert(matrix.getValue((1, 1)) == 2)
    }

    "be updatable" in {
      val m = matrix.updated((0, 0), 7)

      assert(m.size == 6)
      assert(m.getValue((0, 0)) == 7)
    }

    "be additonable with another matrix" in {
      val m = matrix + matrix2
      assertResult(IndexedSeq(3, -1, 3, 6, -2, 8))(m.rowsByColumns)
    }
  }
}
