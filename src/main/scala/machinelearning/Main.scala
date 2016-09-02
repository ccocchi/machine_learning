package machinelearning

import scala.util.Random
import scala.language.implicitConversions
import neuralnetwork.Neuron

object Main {
  def main (args: Array[String]) {
    val or = new Neuron(IndexedSeq(-20, 15, 15))

    println(s"0 OR 0 => ${Math.round(or.compute(new ColVector[Double](IndexedSeq(0.0, 0.0))))}")
    println(s"1 OR 0 => ${Math.round(or.compute(new ColVector[Double](IndexedSeq(1.0, 0.0))))}")
    println(s"0 OR 1 => ${Math.round(or.compute(new ColVector[Double](IndexedSeq(0.0, 1.0))))}")
    println(s"1 OR 1 => ${Math.round(or.compute(new ColVector[Double](IndexedSeq(1.0, 1.0))))}")
  }
}
