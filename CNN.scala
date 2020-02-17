//report2
//畳み込みニューラルネットワークによる画像認識
package report2

import breeze.linalg._

//////////////////////CNN//////////////////////////////
object CNN{
  //-----------実験データ取得--------------
  def load_mnist(dir:String) = {
    def fd(line:String) = line.split(",").map(_.toDouble / 256).toArray
    def ft(line:String) = line.split(",").map(_.toInt).toArray
    val train_d = scala.io.Source.fromFile(dir + "/train-d.txt").getLines.map(fd).toArray
    val train_t = scala.io.Source.fromFile(dir + "/train-t.txt").getLines.map(ft).toArray.head
    val test_d = scala.io.Source.fromFile(dir + "/test-d.txt").getLines.map(fd).toArray
    val test_t = scala.io.Source.fromFile(dir + "/test-t.txt").getLines.map(ft).toArray.head
    (train_d.zip(train_t), test_d.zip(test_t))
  }

  //--------------学習-------------------
  def main(args:Array[String]) {
    val ln = 20 // 学習回数 ★
    val dn = 1000 // 学習データ数 ★
    val tn = 500 // テストデータ数 ★

    // ネットワークの作成 ★
    val layers = List[Layer]()
    //7層
    val con1 = new Convolution(3,28,28,1,20)
    val re1 = new ReLU()
    val po1 = new Pooling(2,20,26,26)
    val con2 = new Convolution(4,13,13,20,20)
    val re2 = new ReLU()
    val po2 = new Pooling(2,20,10,10)
    val af = new Affine(5*5*20,10)
   
    //4層
    val con_ = new Convolution(4,13,13,20,20)
    val re_ = new ReLU()
    val po_ = new Pooling(2,20,10,10)
    val af_ = new Affine(5*5*20,10)
   


    val rand = new scala.util.Random(0)

    for(i<-0 until con1.OC ; j<-0 until con1.IC ; k<-0 until con1.KW ; l<-0 until con1.KW){
      con1.K(i)(j)(k)(l) = rand.nextDouble * 0.01
    }
    for(i<-0 until con2.OC ; j<-0 until con2.IC ; k<-0 until con2.KW ; l<-0 until con2.KW){
      con2.K(i)(j)(k)(l) = rand.nextDouble * 0.01
    }
 
    //-------データの読み込み----------
    val (dtrain,dtest) = load_mnist("/home/share/number")

    //----------０から９までの画像データをまとめる--------
    val ds = (0 until 10).map(i=>dtrain(i*2+1)._1).toArray 

    //-----------学習およびテスト★-------------
    for(i <- 0 until ln) {
      println(i+1 + "<回目>")
      var correct = 0d //正解数
      var err = 0d //平均二乗誤差
      //x:入力、n:正解
      for((x,n)<-rand.shuffle(dtrain.toList)take(dn)) {
        //-------学習---------
        //val y = af.forward(po2.forward(re2.forward(con2.forward(po1.forward(re1.forward(con1.forward(x)))))))
        val y = af_.forward(po_.forward(re_.forward(con_.forward(x))))
        var t = new Array[Double](10)
        t(n) = 1d
        var d = new Array[Double](10)
        for(j<-0 until d.size){
          d(j) = y(j) - t(j)
        }

        //------平均二乗誤差---------
        err += MSE(y,t)

        //con1.backward(re1.backward(po1.backward(con2.backward(re2.backward(po2.backward(af.backward(d)))))))
        con_.backward(re_.backward(po_.backward(af_.backward(d))))

        con1.update()
        re1.update()
        po1.update()
        con2.update()
        re2.update()
        po2.update()
        af.update()

        con_.update()
        re_.update()
        po_.update()
        af_.update()



        if(y.indexOf(y.max) == n){correct += 1d}
      }

      //println("正解率："+ correct/dn*100 + "%")
      println("平均二乗誤差:" + err / dn )

      if(i == 0 || i == ln-1 ){
        Image.write(f"cnn" + i + "-$i%02d.png" , make_image(ds,10,26,26))
      }

      //---------テスト----------
      var correcttest = 0d
      var errtest = 0d
      for((x,n) <- dtest.take(tn)) {
        //val y = af.forward(po2.forward(re2.forward(con2.forward(po1.forward(re1.forward(con1.forward(x)))))))
        val y = af_.forward(po_.forward(re_.forward(con_.forward(x))))

        //------平均二乗誤差---------
        var t = new Array[Double](10)
        t(n) = 1d
        errtest += MSE(y,t)

        con1.reset()
        re1.reset()
        po1.reset()
        con2.reset()
        re2.reset()
        po2.reset()
        af.reset()

        con_.reset()
        re_.reset()
        po_.reset()
        af_.reset()

        if(argmax(y) == n){correcttest += 1d}
        //println(argmax(y),n)
      }
      //println("テスト正解率："+ correcttest/tn*100 + "%")
      println("テスト平均二乗誤差:" + errtest / dn )
    }


    //-----------平均二乗誤差----------------
    //a:出力、b:正解
    def MSE(a:Array[Double] , b:Array[Double])={
      var sum = 0d
      for(i<-0 until a.size){
        sum += (a(i) - b(i)) * (a(i) - b(i))
      }
      sum
    }

    //-------------------出力を画像に変換----------------------------
    def make_image(xs:Array[Array[Double]],C:Int,H:Int,W:Int)={
      val im = Array.ofDim[Int](xs.size*H,C*W,3)
      for(i<-0 until xs.size){
        var y = re1.forward(con1.forward(xs(i)))
        re1.reset()
        con1.reset()
        val ymin = y.min
        val ymax = y.max
        def f(a:Double)=((a-ymin)/(ymax-ymin)*255).toInt
        for(j<-0 until C){
          for(p<-0 until H ; q<-0 until W ; k<-0 until 3){
            im(i*H+p)(j*W+q)(k) = f( y( j*H*W + p*W+q ) )
          }
        }
      }
      im
    }
  } //main
} //CNN

abstract class Layer {
  def forward(x:Array[Double]) : Array[Double]
  def backward(x:Array[Double]) : Array[Double]
  def update() : Unit
  def reset() : Unit
}


/////////////////////畳み込みニューラルネットワーク////////////////////
class Convolution(val KW:Int,val IH:Int,val IW:Int,val IC:Int,val OC:Int) extends Layer {
  //I:入力、O:出力、H:height、W:width、C:チャネル数
  val OH = IH - KW + 1
  val OW = IW - KW + 1
  // 必要なパラメータを定義する ★
  //(セット,枚数目,行,列)
  //カーネル
  val rand = new util.Random(0)
  var K = Array.ofDim[Double](OC,IC,KW,KW)
  for(i<-0 until OC ; j<-0 until IC ; k<-0 until KW ; l<-0 until KW){
    K(i)(j)(k)(l) = rand.nextGaussian() * 0.01
  }
  var dK = Array.ofDim[Double](OC,IC,KW,KW)
  var V_new = Array[Double]()
  var s = Array.ofDim[Double](OC,IC,KW,KW)
  var r = Array.ofDim[Double](OC,IC,KW,KW)
  var p1t = 1d //updataで使うp1のt乗
  var p2t = 1d //updataで使うp2のt乗

  def forward(V:Array[Double]) = {
    V_new = V
    var Z = Array.ofDim[Double](OC * OH * OW)
    // Zを計算する (9.7) ★
    //------------畳み込み-------------------
    for(i<-0 until OC ; j<-0 until OH ; k<-0 until OW){
      for(l<-0 until IC ; m<-0 until KW ; n<-0 until KW){
        Z(i*OH*OW + j*OW+k) += V(l*IH*IW + (j+m)*IW+(k+n)) * K(i)(l)(m)(n)
      }
    }
    Z
  }

  def backward(G:Array[Double]) = {
    // dKを計算する (9.11) ★
    for(i<-0 until OC ; j<-0 until IC ; k<-0 until KW ; l<-0 until KW){
      for(m<-0 until OH ; n<-0 until OW){
        dK(i)(j)(k)(l) += G(i*OH*OW + m*OW+n) * V_new(j*IH*IW + (m+k)*IW+(n+l))
      }
    }
    
    // dVを計算する (9.13) ★
    var dV = Array.ofDim[Double](IC * IH * IW)
    for(i<-0 until IC ; j<-0 until IH ; k<-0 until IW){
      for(l<-0 until OH ; m<-0 until KW){
        if(l + m == j){
          for(n<-0 until OW ; p<-0 until KW){
            if(n + p == k){
              for(q<-0 until OC){
                dV(i*IW*IH + j*IW+k) += K(q)(i)(m)(p)  * G(q*OH*OW + l*OW+n)
              }
            }
          }
        }
      }
    }
    dV
  }

  def update() {
    // Kを更新する(Adam) ★
    val ep = 0.001 //学習率
    val p1 = 0.9 //モーメントの推定に対する指数減衰率
    val p2 = 0.999 //モーメントの推定に対する指数減衰率
    val delta = 0.00000001 //数値安定のための小さな定数デルタ
    
    p1t = p1t * p1 //updata（p1）が何回呼び出されたか
    p2t = p2t * p2

    for(i<-0 until OC ; j<-0 until IC ; k<-0 until KW ; l<-0 until KW){
      s(i)(j)(k)(l) = p1 * s(i)(j)(k)(l) + (1-p1) * dK(i)(j)(k)(l) 
      r(i)(j)(k)(l) = p2 * r(i)(j)(k)(l) + (1-p2) * ( dK(i)(j)(k)(l) * dK(i)(j)(k)(l) )
      val s_ = s(i)(j)(k)(l) / (1-p1t)
      val r_ = r(i)(j)(k)(l) / (1-p2t)
      K(i)(j)(k)(l) = K(i)(j)(k)(l) - ep * s_ / ( math.sqrt( r_ ) + delta )
    }

    reset()
  }

  def reset() {
    // dKを初期化する ★
    dK = Array.ofDim[Double](OC,IC,KW,KW)
  }
}


//////////////////////Affine/////////////////////////////
class Affine(val xn:Int, val yn:Int) extends Layer {
  val rand = new util.Random(0)
  var W = DenseMatrix.zeros[Double](yn,xn).map(_=>rand.nextGaussian * 0.01)
  var b = DenseVector.zeros[Double](yn)

  var dW = DenseMatrix.zeros[Double](yn,xn)
  var db = DenseVector.zeros[Double](yn)

  var xs = List[Array[Double]]()

  var sW = DenseMatrix.zeros[Double](yn,xn)
  var sb = DenseVector.zeros[Double](yn)
  var rW = DenseMatrix.zeros[Double](yn,xn)
  var rb = DenseVector.zeros[Double](yn)

  var p1t = 1d //updataで使うp1のt乗
  var p2t = 1d

  def push(x:Array[Double]) = { xs ::= x; x }
  def pop() = { val x = xs.head; xs = xs.tail; x }

  def forward(x:Array[Double]) = {
    push(x)
    val xv = DenseVector(x)
    val y = W * xv + b
    y.toArray
  }

  def backward(d:Array[Double]) = {
    val x = pop()
    val dv = DenseVector(d)
    // dW,dbを計算する ★
    dW += dv * DenseVector(x).t
    db += dv

    var dx = DenseVector.zeros[Double](xn)
    // dxを計算する ★
    dx = W.t * dv
    dx.toArray
  }

  def update() {
    // W,bを更新する(Adam) ★
    val ep = 0.001 //学習率
    val p1 = 0.9 //モーメントの推定に対する指数減衰率
    val p2 = 0.999 //モーメントの推定に対する指数減衰率
    val delta = 0.00000001 //数値安定のための小さな定数デルタ
   
    p1t = p1t * p1 //updata（p1）が何回呼び出されたか
    p2t = p2t * p2

    sW = p1 * sW + (1-p1) * dW
    sb = p1 * sb + (1-p1) * db
    rW = p2 * rW + (1-p2) * ( dW *:* dW )
    rb = p2 * rb + (1-p2) * ( db *:* db )
    val s_W = sW /:/ (1-p1t)
    val s_b = sb /:/ (1-p1t)
    val r_W = rW /:/ (1-p2t)
    val r_b = rb /:/ (1-p2t)

    W += -ep * s_W / (r_W.map(math.sqrt) + delta )
    b += -ep * s_b / (r_b.map(math.sqrt) + delta )

    reset()
  }

  def reset() {
    dW = DenseMatrix.zeros[Double](yn,xn)
    db = DenseVector.zeros[Double](yn)
    xs = List[Array[Double]]()
  }
}

////////////////////////プーリング/////////////////////////////
class Pooling(val BW:Int, val IC:Int, val IH:Int, val IW:Int) extends Layer {
  val OH = IH / BW
  val OW = IW / BW
  val OC = IC
  var masks = List[Array[Double]]()
  def push(x:Array[Double]) = { masks ::= x; x }
  def pop() = { val mask = masks.head; masks = masks.tail; mask }

  def iindex(i:Int, j:Int, k:Int) = i * IH * IW + j * IW + k
  def oindex(i:Int, j:Int, k:Int) = i * OH * OW + j * OW + k
  
  def forward(X:Array[Double]) = {
    val mask = push(Array.ofDim[Double](IC * IH * IW))
    val Z = Array.ofDim[Double](OC * OH * OW)
    for(i <- 0 until OC; j <- 0 until OH; k <- 0 until OW) {
      var v = Double.NegativeInfinity
      var row_max = -1
      var col_max = -1
      for(m <- 0 until BW; n <- 0 until BW if v < X(iindex(i,j*BW+m,k*BW+n))) {
        row_max = j*BW+m
        col_max = k*BW+n
        v = X(iindex(i,j*BW+m,k*BW+n))
      }
      mask(iindex(i,row_max,col_max)) = 1
      Z(oindex(i,j,k)) = v
    }
    Z
  }

  def backward(d:Array[Double]) = {
    val mask = pop()
    val D = Array.ofDim[Double](mask.size)
    for(i <- 0 until OC; j <- 0 until OH; k <- 0 until OW) {
      for(m <- 0 until BW; n <- 0 until BW if mask(iindex(i,j*BW+m,k*BW+n)) == 1) {
        D(iindex(i,j*BW+m,k*BW+n)) = d(oindex(i,j,k))
      }
    }
    D
  }

  def update() {
    reset()
  }

  def reset() {
    masks = List[Array[Double]]()
  }
}

//////////////////////ReLU/////////////////////////////
class ReLU() extends Layer {
  var ys = List[Array[Double]]()
  def push(y:Array[Double]) = { ys ::= y; y }
  def pop() = { val y = ys.head; ys = ys.tail; y }

  def forward(x:Array[Double]) = {
    push(x.map(a => math.max(a,0)))
  }

  def backward(d:Array[Double]) = {
    val y = pop()
    (0 until d.size).map(i => if(y(i) > 0) d(i) else 0d).toArray
  }

  def update() {
    reset()
  }

  def reset() {
    ys = List[Array[Double]]()
  }
}



object Image {
  def rgb(im : java.awt.image.BufferedImage, i:Int, j:Int) = {
    val c = im.getRGB(i,j)
    Array(c >> 16 & 0xff, c >> 8 & 0xff, c & 0xff)
  }

  def pixel(r:Int, g:Int, b:Int) = {
    val a = 0xff
    ((a & 0xff) << 24) | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff)
  }

  def read(fn:String) = {
    val im = javax.imageio.ImageIO.read(new java.io.File(fn))
    (for(i <- 0 until im.getHeight; j <- 0 until im.getWidth) 
      yield rgb(im, j, i)).toArray.grouped(im.getWidth).toArray
  }

  def write(fn:String, b:Array[Array[Array[Int]]]) = {
    val w = b(0).size
    val h = b.size
    val im = new java.awt.image.BufferedImage(w, h, java.awt.image.BufferedImage.TYPE_INT_RGB);
    for(i <- 0 until im.getHeight; j <- 0 until im.getWidth) {
      im.setRGB(j,i,pixel(b(i)(j)(0), b(i)(j)(1), b(i)(j)(2)));
    }
    javax.imageio.ImageIO.write(im, "png", new java.io.File(fn))
  }
}





















