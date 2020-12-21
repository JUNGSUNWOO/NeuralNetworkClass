**신경망과 딥러닝 CNN 실습 프로젝트**

![학습 과정](D:\딥러닝자료\학습 과정.jpg)



**데이터 전처리 **

IMDB, WIKI face_crop dataset

![IMDB_CROP](D:\딥러닝자료\IMDB_CROP.JPG)

![WIKI_CROP](D:\딥러닝자료\WIKI_CROP.JPG)

다운받은 imdb, wiki 의 face crop 데이터들을 labeling되어있는 mat파일을 이용해 분류해서 저장해준다

![image-20201220185222359](C:\Users\정선우\AppData\Roaming\Typora\typora-user-images\image-20201220185222359.png)

gender 폴더 안에는 male, female 로 구분되어 사진들이 저장되어있다.

*사진들중 사람이 아닌 것 혹은 학습에 불리하게 작용될 data들을 사전 점검하여 지워준다



CNN을 이용한 classification model을 구현



transforms = [resize : 256 x 256, crop : 224 x 224, grascale : true, toTensor, normalize(0.5),(0.5)]

hyper-parameters

epochs, batch_size, learning_rate, momentum, network_layer



Forward

활성함수 : ReLU

Network 구조 : Conv2d + maxpooling Layer 5층, Linear Layer 3층의 Network



Backward

CrossEntropyLoss(교차엔트로피 손실함수)

SGD(**Stochastic Gradient Descent**)(경사하강법)

- 미니배치 단위로 갱신 함 계산량이 적어서 배치 전체 단위보다 빠름



실습

1. image resize, grayscale 등은 사용하지 않았다. 1 Epoch 만으로도 굉장히 긴 시간을 요구했음

1-Epoch train

Epoch [1/1], Step [10000/42797], Acc : 64.6900, Loss: 0.5259
Epoch [1/1], Step [20000/42797], Acc : 69.5000, Loss: 0.6746
Epoch [1/1], Step [30000/42797], Acc : 72.5900, Loss: 0.3590
Epoch [1/1], Step [40000/42797], Acc : 75.0713, Loss: 0.3820
Finished Training
Accuracy of the network: 88.97193126327953 %
Accuracy of woman: 75.27737073477077 %
Accuracy of man: 91.65117228243616 %



2. image resize, grayscale 등을 사용하였다. 결론적으로 학습시간은 월등이 빨라졌지만 학습결과는 여전히 망했다.

5-Epoch train

Epoch [1/5], Step [10000/42797], Acc : 76.0700, Loss: 0.1816
Epoch [1/5], Step [20000/42797], Acc : 80.3525, Loss: 0.6877
Epoch [1/5], Step [30000/42797], Acc : 82.2342, Loss: 0.4017
Epoch [1/5], Step [40000/42797], Acc : 83.2794, Loss: 0.1918
Epoch [2/5], Step [10000/42797], Acc : 84.1921, Loss: 0.3172
Epoch [2/5], Step [20000/42797], Acc : 84.6827, Loss: 0.6108
Epoch [2/5], Step [30000/42797], Acc : 85.0649, Loss: 0.0158
Epoch [2/5], Step [40000/42797], Acc : 85.4330, Loss: 0.2178
Epoch [3/5], Step [10000/42797], Acc : 85.8588, Loss: 0.4152
Epoch [3/5], Step [20000/42797], Acc : 86.1274, Loss: 0.4360
Epoch [3/5], Step [30000/42797], Acc : 86.3335, Loss: 0.1576
Epoch [3/5], Step [40000/42797], Acc : 86.5229, Loss: 0.0255
Epoch [4/5], Step [10000/42797], Acc : 86.8014, Loss: 0.9058
Epoch [4/5], Step [20000/42797], Acc : 86.9649, Loss: 0.0186
Epoch [4/5], Step [30000/42797], Acc : 87.1099, Loss: 0.3215
Epoch [4/5], Step [40000/42797], Acc : 87.2341, Loss: 0.0264
Epoch [5/5], Step [10000/42797], Acc : 87.4514, Loss: 0.1884
Epoch [5/5], Step [20000/42797], Acc : 87.6058, Loss: 0.2444
Epoch [5/5], Step [30000/42797], Acc : 87.7153, Loss: 0.1248
Epoch [5/5], Step [40000/42797], Acc : 87.8250, Loss: 0.0780
Finished Training
Accuracy of the network: 88.474298281582 %
Accuracy of woman: 84.67289719626169 %
Accuracy of man: 89.55162858310997 %



이것을 제외하고도 epoch, learning_rate, momentum, network layer등 다양한 하이퍼파라미터등을 조정하였지만,

Loss 가 계속 왔다갔다하며 떨어지는 방향으로 수렴하지 못하였다.

그렇게 계속 학습이 잘 되지 않았다.

이유가 무엇일지 고민을 하다가

데이터셋에 대해서 고민을 하였다. 나는 남, 녀로 분류할거를 알고있지만 컴퓨터는 모르기 때문에 CNN네트워크를 통과하더라도 얼굴 이외의 잡다한 것 들은 무쓸모한 정보이다.

데이터 transform과정에서 resize 뿐만아니라 centercrop을 통하여 최대한 얼굴 부분만 네트워크에 입력 되도록 하였다. 



3. image resize & image crop을 통하여 최대한 얼굴 부분을 네트워크로 입력하였다.

4. dataset에서 train set에 적합하지 않은 헷갈릴 법한 데이터들의 전처리를 다시 해주었다

   

   Epoch 20회 - 시간을 줄이고자 mini dataset을 만들어서 학습 함

   Epoch [1/20], Step [500/546], Acc: [52.6500], Loss: 0.6764
   Epoch [2/20], Step [500/546], Acc: [54.8736], Loss: 0.6702
   Epoch [3/20], Step [500/546], Acc: [58.8864], Loss: 0.5036
   Epoch [4/20], Step [500/546], Acc: [62.5749], Loss: 0.5645
   Epoch [5/20], Step [500/546], Acc: [65.5944], Loss: 0.5733
   Epoch [6/20], Step [500/546], Acc: [68.1196], Loss: 0.3139
   Epoch [7/20], Step [500/546], Acc: [70.2242], Loss: 0.5141
   Epoch [8/20], Step [500/546], Acc: [72.1297], Loss: 0.3499
   Epoch [9/20], Step [500/546], Acc: [73.7786], Loss: 0.2721
   Epoch [10/20], Step [500/546], Acc: [75.2463], Loss: 0.2535
   Epoch [11/20], Step [500/546], Acc: [76.6118], Loss: 0.1687
   Epoch [12/20], Step [500/546], Acc: [77.8701], Loss: 0.0786
   Epoch [13/20], Step [500/546], Acc: [78.9922], Loss: 0.3576
   Epoch [14/20], Step [500/546], Acc: [80.1241], Loss: 0.1604
   Epoch [15/20], Step [500/546], Acc: [81.1204], Loss: 0.0035
   Epoch [16/20], Step [500/546], Acc: [82.0468], Loss: 0.1627
   Epoch [17/20], Step [500/546], Acc: [82.9152], Loss: 0.0308
   Epoch [18/20], Step [500/546], Acc: [83.7078], Loss: 0.0843
   Epoch [19/20], Step [500/546], Acc: [84.4643], Loss: 0.0090
   Epoch [20/20], Step [500/546], Acc: [85.1282], Loss: 0.0494
   Finished Training
   Accuracy of the network: 79.13476263399694 %
   Accuracy of woman: 86.25592417061611 %
   Accuracy of man: 73.5042735042735 %



5. loss 가 최종적으로는 떨어졌지만 떨어지는 과정이 순탄치 않음 정확도는 80%정도

6. 다음 단계에서는 이제 다시 하이퍼 파라미터를 조정해보겠음. 

   learning_rate 0.01, momentum = x, epoch = 20, batch_size = 32

Epoch [1/20], Step [500/1092], Loss: 0.6594
Epoch [1/20], Step [1000/1092], Loss: 0.7188
Epoch [2/20], Step [500/1092], Loss: 0.7075
Epoch [2/20], Step [1000/1092], Loss: 0.7086
Epoch [3/20], Step [500/1092], Loss: 0.6945
Epoch [3/20], Step [1000/1092], Loss: 0.7180
Epoch [4/20], Step [500/1092], Loss: 0.6962
Epoch [4/20], Step [1000/1092], Loss: 0.6800
Epoch [5/20], Step [500/1092], Loss: 0.6702
Epoch [5/20], Step [1000/1092], Loss: 0.6795
Epoch [6/20], Step [500/1092], Loss: 0.7128
Epoch [6/20], Step [1000/1092], Loss: 0.7117
Epoch [7/20], Step [500/1092], Loss: 0.5808
Epoch [7/20], Step [1000/1092], Loss: 0.6348
Epoch [8/20], Step [500/1092], Loss: 0.2834
Epoch [8/20], Step [1000/1092], Loss: 0.5217
Epoch [9/20], Step [500/1092], Loss: 0.3944
Epoch [9/20], Step [1000/1092], Loss: 0.5888
Epoch [10/20], Step [500/1092], Loss: 0.3320
Epoch [10/20], Step [1000/1092], Loss: 0.4647
Epoch [11/20], Step [500/1092], Loss: 0.3170
Epoch [11/20], Step [1000/1092], Loss: 0.4201
Epoch [12/20], Step [500/1092], Loss: 0.2691
Epoch [12/20], Step [1000/1092], Loss: 0.0679
Epoch [13/20], Step [500/1092], Loss: 0.4617
Epoch [13/20], Step [1000/1092], Loss: 0.4652
Epoch [14/20], Step [500/1092], Loss: 0.6078
Epoch [14/20], Step [1000/1092], Loss: 0.2011
Epoch [15/20], Step [500/1092], Loss: 0.3705
Epoch [15/20], Step [1000/1092], Loss: 0.5708
Epoch [16/20], Step [500/1092], Loss: 0.2597
Epoch [16/20], Step [1000/1092], Loss: 0.0618
Epoch [17/20], Step [500/1092], Loss: 0.1993
Epoch [17/20], Step [1000/1092], Loss: 0.3596
Epoch [18/20], Step [500/1092], Loss: 0.5330
Epoch [18/20], Step [1000/1092], Loss: 0.2494
Epoch [19/20], Step [500/1092], Loss: 0.0997
Epoch [19/20], Step [1000/1092], Loss: 0.0866
Epoch [20/20], Step [500/1092], Loss: 0.2515
Epoch [20/20], Step [1000/1092], Loss: 0.0425
Finished Training
Accuracy of the network: 79.13476263399694 %
Accuracy of woman: 85.37735849056604 %
Accuracy of man: 72.17391304347827 %

![epoch 20graph](D:\딥러닝자료\epoch 20graph.png)

7. https://github.com/oarriaga/face_classification/blob/master/report.pdf

   위의 논문을 참조하여 CNN의 네트워크 구조를 따라가보았다

   위의 논문에서는 MAXPOOLING 과 BATCH NORMALIZATION을 사용하였으며

   Conv2d - BatchNorm x 2 -> 1. Conv2d - BatchNorm x 2 - MaxPool =>

   ​												2. Conv2d - BatchNorm =>

=> 1+2-> Conv2d - Avg Pooling - SoftMax 로 구성되어있었다

본인 또한 위의 논문의 네트워크 구조를 따라 Batch Normalization 과 pooling을 함께 사용하였으며, 결과는 이전의 네트워크 구조보다 좀더 안정적으로 Loss가 떨어졌으며 Accuracy 가 매우 높아진 것을 확인하였다.

여기서 pooling과 batchnormalization이 뭐길래 이렇게 결과가 달라진지 궁금하게 되었고 이에 대한 공부를 했다.



어렵지않게 배치 정규화가 CNN의 중요한 기술중 하나임을 알 수 있었다.

Batch Normalization 이란 딥러닝에서 입력의 절대값이 비선형 포화함수(sigmoid와 같은)에서 작은 일부 구간을 제외하면 미분값이 0 근처로 가기 때문에 역전파를 통한 학습이 어려워 지거나 느려지게 된다.

ReLU를 활성함수로 쓰면 문제가 완화되긴 하지만 간접적인 회피이기 때문에 layer가 깊어질수록 문제가 된다. 

Batch Normalization이 무엇일까?

normalization은 기존에도 training 전체 집합에 대하여 실시해왔다. mini-batch SGD방식ㅇ르 사용ㅎ



Epoch [1/20], Step [100/546], Loss: 0.6684
Epoch [1/20], Step [200/546], Loss: 0.6108
Epoch [1/20], Step [300/546], Loss: 0.6188
Epoch [1/20], Step [400/546], Loss: 0.4560
Epoch [1/20], Step [500/546], Loss: 0.4985
Epoch [2/20], Step [100/546], Loss: 0.6051
Epoch [2/20], Step [200/546], Loss: 0.5734
Epoch [2/20], Step [300/546], Loss: 0.3348
Epoch [2/20], Step [400/546], Loss: 0.4005
Epoch [2/20], Step [500/546], Loss: 0.3566
Epoch [3/20], Step [100/546], Loss: 0.2690
Epoch [3/20], Step [200/546], Loss: 0.8305
Epoch [3/20], Step [300/546], Loss: 0.2927
Epoch [3/20], Step [400/546], Loss: 0.2234
Epoch [3/20], Step [500/546], Loss: 0.6410
Epoch [4/20], Step [100/546], Loss: 0.5124
Epoch [4/20], Step [200/546], Loss: 0.3981
Epoch [4/20], Step [300/546], Loss: 0.4500
Epoch [4/20], Step [400/546], Loss: 0.2639
Epoch [4/20], Step [500/546], Loss: 0.5393
Epoch [5/20], Step [100/546], Loss: 0.1768
Epoch [5/20], Step [200/546], Loss: 0.2734
Epoch [5/20], Step [300/546], Loss: 0.1863
Epoch [5/20], Step [400/546], Loss: 0.2884
Epoch [5/20], Step [500/546], Loss: 0.3688
Epoch [6/20], Step [100/546], Loss: 0.0954
Epoch [6/20], Step [200/546], Loss: 0.1172
Epoch [6/20], Step [300/546], Loss: 0.7088
Epoch [6/20], Step [400/546], Loss: 0.1372
Epoch [6/20], Step [500/546], Loss: 0.0948
Epoch [7/20], Step [100/546], Loss: 0.0957
Epoch [7/20], Step [200/546], Loss: 0.1294
Epoch [7/20], Step [300/546], Loss: 0.0533
Epoch [7/20], Step [400/546], Loss: 0.0883
Epoch [7/20], Step [500/546], Loss: 0.1174
Epoch [8/20], Step [100/546], Loss: 0.1059
Epoch [8/20], Step [200/546], Loss: 0.0383
Epoch [8/20], Step [300/546], Loss: 0.0837
Epoch [8/20], Step [400/546], Loss: 0.0372
Epoch [8/20], Step [500/546], Loss: 0.0306
Epoch [9/20], Step [100/546], Loss: 0.0152
Epoch [9/20], Step [200/546], Loss: 0.0167
Epoch [9/20], Step [300/546], Loss: 0.0630
Epoch [9/20], Step [400/546], Loss: 0.0135
Epoch [9/20], Step [500/546], Loss: 0.0012
Epoch [10/20], Step [100/546], Loss: 0.2395
Epoch [10/20], Step [200/546], Loss: 0.0123
Epoch [10/20], Step [300/546], Loss: 0.0132
Epoch [10/20], Step [400/546], Loss: 0.0107
Epoch [10/20], Step [500/546], Loss: 0.0017
Epoch [11/20], Step [100/546], Loss: 0.0342
Epoch [11/20], Step [200/546], Loss: 0.0041
Epoch [11/20], Step [300/546], Loss: 0.0015
Epoch [11/20], Step [400/546], Loss: 0.0250
Epoch [11/20], Step [500/546], Loss: 0.0034
Epoch [12/20], Step [100/546], Loss: 0.0075
Epoch [12/20], Step [200/546], Loss: 0.0010
Epoch [12/20], Step [300/546], Loss: 0.0101
Epoch [12/20], Step [400/546], Loss: 0.1699
Epoch [12/20], Step [500/546], Loss: 0.0021
Epoch [13/20], Step [100/546], Loss: 0.0016
Epoch [13/20], Step [200/546], Loss: 0.0004
Epoch [13/20], Step [300/546], Loss: 0.0031
Epoch [13/20], Step [400/546], Loss: 0.0013
Epoch [13/20], Step [500/546], Loss: 0.0690
Epoch [14/20], Step [100/546], Loss: 0.1711
Epoch [14/20], Step [200/546], Loss: 0.0012
Epoch [14/20], Step [300/546], Loss: 0.0026
Epoch [14/20], Step [400/546], Loss: 0.0025
Epoch [14/20], Step [500/546], Loss: 0.1277
Epoch [15/20], Step [100/546], Loss: 0.0229
Epoch [15/20], Step [200/546], Loss: 0.0022
Epoch [15/20], Step [300/546], Loss: 0.0021
Epoch [15/20], Step [400/546], Loss: 0.0010
Epoch [15/20], Step [500/546], Loss: 0.0014
Epoch [16/20], Step [100/546], Loss: 0.0014
Epoch [16/20], Step [200/546], Loss: 0.0067
Epoch [16/20], Step [300/546], Loss: 0.0016
Epoch [16/20], Step [400/546], Loss: 0.0017
Epoch [16/20], Step [500/546], Loss: 0.0027
Epoch [17/20], Step [100/546], Loss: 0.0007
Epoch [17/20], Step [200/546], Loss: 0.0014
Epoch [17/20], Step [300/546], Loss: 0.0008
Epoch [17/20], Step [400/546], Loss: 0.0011
Epoch [17/20], Step [500/546], Loss: 0.0807
Epoch [18/20], Step [100/546], Loss: 0.0002
Epoch [18/20], Step [200/546], Loss: 0.0022
Epoch [18/20], Step [300/546], Loss: 0.0062
Epoch [18/20], Step [400/546], Loss: 0.0009
Epoch [18/20], Step [500/546], Loss: 0.0009
Epoch [19/20], Step [100/546], Loss: 0.0006
Epoch [19/20], Step [200/546], Loss: 0.0711
Epoch [19/20], Step [300/546], Loss: 0.0014
Epoch [19/20], Step [400/546], Loss: 0.0040
Epoch [19/20], Step [500/546], Loss: 0.0008
Epoch [20/20], Step [100/546], Loss: 0.0016
Epoch [20/20], Step [200/546], Loss: 0.0006
Epoch [20/20], Step [300/546], Loss: 0.0006
Epoch [20/20], Step [400/546], Loss: 0.0010
Epoch [20/20], Step [500/546], Loss: 0.0010
Finished Training
Accuracy of the network: 76.33996937212864 %
Accuracy of woman: 79.90867579908675 %
Accuracy of man: 80.73394495412845 %

Accuracy는 크게 변화 없지만, Loss는 최종적으로 떨어지는 그래프를 그림을 확인 할 수 있다.





Epoch [1/15], Step [200/546], Acc: 59.9062, Loss: 0.7134
Epoch [1/15], Step [400/546], Acc: 63.5000, Loss: 0.5298
Epoch [2/15], Step [200/546], Acc: 67.2421, Loss: 0.6072
Epoch [2/15], Step [400/546], Acc: 68.5257, Loss: 0.4450
Epoch [3/15], Step [200/546], Acc: 70.3910, Loss: 0.4047
Epoch [3/15], Step [400/546], Acc: 71.1975, Loss: 0.2889
Epoch [4/15], Step [200/546], Acc: 72.6419, Loss: 0.4364
Epoch [4/15], Step [400/546], Acc: 73.3581, Loss: 0.3271
Epoch [5/15], Step [200/546], Acc: 74.6565, Loss: 0.1722
Epoch [5/15], Step [400/546], Acc: 75.1936, Loss: 0.3920
Epoch [6/15], Step [200/546], Acc: 76.1528, Loss: 0.2145
Epoch [6/15], Step [400/546], Acc: 76.6184, Loss: 0.2379
Epoch [7/15], Step [200/546], Acc: 77.4488, Loss: 0.3081
Epoch [7/15], Step [400/546], Acc: 77.9771, Loss: 0.1669
Epoch [8/15], Step [200/546], Acc: 78.8096, Loss: 0.3022
Epoch [8/15], Step [400/546], Acc: 79.3117, Loss: 0.3091
Epoch [9/15], Step [200/546], Acc: 80.1516, Loss: 0.1786
Epoch [9/15], Step [400/546], Acc: 80.6160, Loss: 0.1501
Epoch [10/15], Step [200/546], Acc: 81.4443, Loss: 0.2599
Epoch [10/15], Step [400/546], Acc: 81.9064, Loss: 0.5269
Epoch [11/15], Step [200/546], Acc: 82.7273, Loss: 0.1118
Epoch [11/15], Step [400/546], Acc: 83.1719, Loss: 0.1633
Epoch [12/15], Step [200/546], Acc: 83.9306, Loss: 0.0778
Epoch [12/15], Step [400/546], Acc: 84.3309, Loss: 0.0431
Epoch [13/15], Step [200/546], Acc: 85.0532, Loss: 0.0120
Epoch [13/15], Step [400/546], Acc: 85.4230, Loss: 0.2008
Epoch [14/15], Step [200/546], Acc: 86.0480, Loss: 0.0456
Epoch [14/15], Step [400/546], Acc: 86.3986, Loss: 0.0538
Epoch [15/15], Step [200/546], Acc: 86.9419, Loss: 0.0083
Epoch [15/15], Step [400/546], Acc: 87.2526, Loss: 0.0179
Finished Training
Accuracy of the network: 79.70903522205207 %
Accuracy of woman: 90.0990099009901 %
Accuracy of man: 74.60317460317461 %





Epoch [1/20], Step [200/546], Acc: 63.2188, Loss: 0.6698
Epoch [1/20], Step [400/546], Acc: 65.5156, Loss: 0.4294
Epoch [2/20], Step [200/546], Acc: 68.2393, Loss: 0.5775
Epoch [2/20], Step [400/546], Acc: 69.3451, Loss: 0.4390
Epoch [3/20], Step [200/546], Acc: 71.4410, Loss: 0.6259
Epoch [3/20], Step [400/546], Acc: 72.2827, Loss: 0.3788
Epoch [4/20], Step [200/546], Acc: 73.7576, Loss: 0.7841
Epoch [4/20], Step [400/546], Acc: 74.4992, Loss: 0.1436
Epoch [5/20], Step [200/546], Acc: 75.6425, Loss: 0.2902
Epoch [5/20], Step [400/546], Acc: 76.3041, Loss: 0.5607
Epoch [6/20], Step [200/546], Acc: 77.3669, Loss: 0.4317
Epoch [6/20], Step [400/546], Acc: 77.8967, Loss: 0.3888
Epoch [7/20], Step [200/546], Acc: 78.8877, Loss: 0.3441
Epoch [7/20], Step [400/546], Acc: 79.4364, Loss: 0.1897
Epoch [8/20], Step [200/546], Acc: 80.3174, Loss: 0.3052
Epoch [8/20], Step [400/546], Acc: 80.8562, Loss: 0.1244
Epoch [9/20], Step [200/546], Acc: 81.7106, Loss: 0.1017
Epoch [9/20], Step [400/546], Acc: 82.2144, Loss: 0.0954
Epoch [10/20], Step [200/546], Acc: 83.0947, Loss: 0.1043
Epoch [10/20], Step [400/546], Acc: 83.5559, Loss: 0.0316
Epoch [11/20], Step [200/546], Acc: 84.3908, Loss: 0.2405
Epoch [11/20], Step [400/546], Acc: 84.8437, Loss: 0.0474
Epoch [12/20], Step [200/546], Acc: 85.5666, Loss: 0.0085
Epoch [12/20], Step [400/546], Acc: 85.9784, Loss: 0.0434
Epoch [13/20], Step [200/546], Acc: 86.6319, Loss: 0.0033
Epoch [13/20], Step [400/546], Acc: 86.9878, Loss: 0.0061
Epoch [14/20], Step [200/546], Acc: 87.5498, Loss: 0.0031
Epoch [14/20], Step [400/546], Acc: 87.8536, Loss: 0.0236
Epoch [15/20], Step [200/546], Acc: 88.3630, Loss: 0.0149
Epoch [15/20], Step [400/546], Acc: 88.6400, Loss: 0.2487
Epoch [16/20], Step [200/546], Acc: 89.0845, Loss: 0.0084
Epoch [16/20], Step [400/546], Acc: 89.3206, Loss: 0.3024
Epoch [17/20], Step [200/546], Acc: 89.7200, Loss: 0.0047
Epoch [17/20], Step [400/546], Acc: 89.9328, Loss: 0.0097
Epoch [18/20], Step [200/546], Acc: 90.2843, Loss: 0.0022
Epoch [18/20], Step [400/546], Acc: 90.4773, Loss: 0.0075
Epoch [19/20], Step [200/546], Acc: 90.7927, Loss: 0.0055
Epoch [19/20], Step [400/546], Acc: 90.9630, Loss: 0.0017
Epoch [20/20], Step [200/546], Acc: 91.2421, Loss: 0.0011
Epoch [20/20], Step [400/546], Acc: 91.3995, Loss: 0.0186
Finished Training
Accuracy of the network: 78.4073506891271 %
Accuracy of woman: 78.19905213270142 %
Accuracy of man: 79.48717948717949 %