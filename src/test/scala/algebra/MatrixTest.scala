package algebra

import org.scalatest.WordSpec

class MatrixTest extends WordSpec {
  /*
    [ 1   0
      3   2
      -3  7 ]
   */
  val m1 = IndexedSeq(1, 3, -3, 0, 2, 7).reshape(3, 2)

  "A matrix" can {
    "be accessed randomly" in {
      assertResult(7)(m1(2, 1))
    }

    "be transposed" in {
      val res = m1.transpose
      assert(res.isTranspose)

      assertResult(3)(res.colSize)
      assertResult(2)(res.rowSize)

      assertResult(3)(res(0, 1))
      assertResult(7)(res(1, 2))
    }

//    "be summed by its columns" in {
//      val res = m1.sumColumns
//      assertResult(3)(res.rowSize)
//      assertResult(1)(res.colSize)
//      assertResult(IndexedSeq(1, 5, 4))(res.values)
//    }
  }

  "A vector" can {

  }
}
