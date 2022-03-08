# Facial Recognition System
2021 嵌入式系統期末專題


## 相關套件與工具

### Development Board
Embedsky E9V3

### Dlib
Dlib 是一套使用 C++開發而成的函式庫，內含機器學習的演算法，它被廣泛運用在業界與學術界，包括機器學習、嵌入式系統、移動設備等等，除了開源且免費外，還可以跨平台使用，並提供 Python API。

### OpenCV
OpenCV（Open Source Computer Vision）是一個跨平台的電腦視覺庫，包含許多電腦視覺相關演算處理的 Open Source Library，而且支援多種開法語言。能夠快速的實現一些影像處理和識別的任務。

## 擷取臉部圖片

### 擷取過程
我們的嵌入式系統臉部辨識主要是使用OpenCV 搭配 Dlib 這個套件，原本是想使用dlib_68 點模型來進行臉部圖片偵測，但後來發現在板子上由於運算能力的問題會導致辨識時間太長，故我們犧牲一點精準度改為使用dlib_5點模型來做臉部偵測，該模型可以從一張圖片中辨識出多張臉部的位置(如果有)，以便做之後的人臉身份判別。且5點模型裡可用0~3這幾個點來辨識出眼睛的部分，利於我們框出臉與眼睛。


![](https://i.imgur.com/XaDWqqT.png)
(來源：CH.Tseng - Face Landmark & Alignment)


## 身份識別

### 分群模型
當我們從原片中萃取出臉部的圖片後，我們會用自己訓練的DNN分群模型來給每張臉部圖片一組向量，分群模型裡有3個群，分別為(組員1、組員2、unknow)，得到每張臉部圖片的向量後，我們會與先前已設定好的組員向量做對比，取得臉部圖片與每位組員的距離，再用閥值來判斷是否為組員或是unknow。

### 改善模型
原本分類模型裡只有兩位組員，但發現有時unknow的判斷會有誤，參考過去他人的做法後，在模型裡加入unknow這個cluster可以大大堤升準確度，其中，unknown是以亞洲人臉作為訓練集(AFAD Dataset)。

## 實驗與結果

### 流程圖
![](https://i.imgur.com/NQWqsYP.png)

### 結果
我們的辨識系統在照片清楚的情況下可以精準的判斷出原圖片裡的每張臉部，並識別其身份，但由於Dlib模型與我們訓練模型都是使用正臉來訓練，所以無法辨識側臉。


![](https://i.imgur.com/qbw42b9.png)


## 參考文獻
1.	Website, [Dlib C++ Library - dlib.net](http://dlib.net/)
2.	GitHub, [Dlib C++ library](https://github.com/davisking/dlib)
3.	Website, [OpenCV: Home](https://opencv.org/)
4.	Website, [(Faster) Facial landmark detector with dlib](https://www.pyimagesearch.com/2018/04/02/faster-facial-landmark-detector-with-dlib/)
5.	GitHub, [The AFAD Dataset](https://github.com/afad-dataset/tarball-lite)