# Ultrasonic texture synthesis 超声纹理图像合成
Comparison of different methods - projet inter-disciplinaire de FISE3 Télécom St-Etienne 基于多种合成方法的对比
</br>
</br>
In this repo we try to compare different methods of ultrasonic texture synthesis. We tested Method Efros-Leung(Pixel copy), Method Optim, Method Image-Quilting and Method DeepTexture.
</br>
在这个项目中我们学习了四种主流的纹理合成算法，并测试了它们在超声纹理合成领域的效果，而后设计了GUI以便更加直观的进行对比。
</br>
</br>
The comparison metrics we choose are GIST feature, LBP feature(circular rotation invariant LBP) and SSIM feature.
</br>
我们选择了三种相似度对比的方法来比较合成纹理与源纹理的差异。
</br>
## simple introduction
test.py is the main program which provides a graphical interface to compare the results of different methods.
</br>
</br>
To launch the comparison, first you have to choose an input image in the file "samples" where we provides 15 examples of ultrasound texture images. Their corresponding output images are pre-generated and stored in the file "image_results".
</br>
</br>
Then you have to choose a comparison metric and confirm.
</br>
</br>
Finally you will see the results of each method and the similarity.
## source codes of texture synthesis methods
* Method Pixel Copy </br>
Please refer to [this repo](https://github.com/asteroidhouse/texturesynth)

* Method Optim </br>
Please refer to [this repo](https://github.com/wang-ps/TextureSynthesis)

* Method Quilting </br>
Please refer to [this repo](https://github.com/PJunhyuk/ImageQuilting)

* Method DeepTexture </br>
Please refer to [this repo](https://github.com/meet-minimalist/Texture-Synthesis-Using-Convolutional-Neural-Networks)
## explanation of comparison methods
* GIST </br>
[description](http://ilab.usc.edu/siagian/Research/Gist/Gist.html) in English
</br>
[description](https://zhuanlan.zhihu.com/p/51173086) in Chinese

* LBP(circular and rotation invariant)</br>
[An article](https://www.google.com.hk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=7&ved=2ahUKEwiTw6j56cLmAhWLFcAKHdbJDUEQFjAGegQICRAC&url=http%3A%2F%2Fwww4.comp.polyu.edu.hk%2F~cslzhang%2Fpaper%2FPR_10_Mar_LBPV.pdf&usg=AOvVaw2kyKRaXwuBBHsWpyP8Qst_)precisely explains the LBP feature
</br>
[description](https://blog.csdn.net/zouxy09/article/details/7929531) in Chinese

