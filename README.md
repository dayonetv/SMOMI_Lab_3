# Исследование влияние параметра “темп обучения” на процесс обучения нейронной сети на примере решения задачи классификации Oregon Wildlife с использованием техники обучения Transfer Learning  
Вданной лабораторной работе для решения задачи классификации изображений Oregon Wildlife использовалась нейронная сеть EfficientNet-B0, причем данная будет иметь уже предобученные веса на базе изображений ImageNet. Для того, чтобы наши веса были предобученными на базе изображений ImageNet, необходимо параметр weights выставить равным "imagenet". Также убирается классификатор данной нейронной сети с помощью параметра include_top=False, и создается собственный классификатор (один слой Flatten и полносвязный Dense слой). С помощью model.trainable = False мы замораживаем ту часть нейронной сети которая отвечает за выделение каких-то характерных признаков в изображении так как эта часть уже обучена:  
```
 inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))  
  model = EfficientNetB0(include_top=False, input_tensor=inputs,pooling = 'avg', weights='imagenet')  
  model.trainable = False  
  model = tf.keras.layers.Flatten()(model.output)  
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.softmax)(model)  
  return tf.keras.Model(inputs=inputs, outputs=outputs)  
  ```
В процессе лабораторной работы изменялся темп обучения от 0.1 до 0.0001. Данное изменеие позволит выявить более оптимальный темп обучения для решения задачи классификации изображений Oregon Wildlife исходя из графиков метрики точности, графиков функции потерь и скорости обучения:
```
#optimizer=tf.optimizers.Adam(lr=0.1)
#optimizer=tf.optimizers.Adam(lr=0.01)
#optimizer=tf.optimizers.Adam(lr=0.001)
#optimizer=tf.optimizers.Adam(lr=0.0001)
```
### В результате обучения нейронной сети EfficientNet-B0 (предобученной) с разными темпами обучения (0.1 0.01 0.001 0.0001) получили следующие графики:  
Синяя линия - на валидации  
Оранжевая линия - на обучении  
* График метрики точности для предобученной нейронной сети EfficientNet-B0 с различными темпами обучения (0.1 0.01 0.001 0.0001): 
![legend_accuracy](https://user-images.githubusercontent.com/59259102/111492129-77f97200-874d-11eb-963e-ec33367dd4a5.jpg) 

<img src="./epoch_categorical_accuracy_for_different_lr.svg">

* График функции потерь для предобученной нейронной сети EfficientNet-B0 с различными темпами обучения (0.1 0.01 0.001 0.0001):  

![legend_loss](https://user-images.githubusercontent.com/59259102/111492201-86e02480-874d-11eb-8213-10a1a49438bb.jpg)

<img src="./epoch_loss_for_different_lr_1.svg">

<img src="./epoch_loss_for_different_lr.svg">

* Анализ полученных результатов  
Исходя из полученыых результатов можно отметить следующее: 

### В результате обучения нейронной сети EfficientNet-B0 (предобученной) с разными темпами обучения (0.1 0.01 0.001 0.0001) получили следующие графики:  
Синяя линия - на валидации  
Оранжевая линия - на обучении  
* График метрики точности для нейронной сети EfficientNet-B0 (случайное начальное приближение):  
<img src="./epoch_categorical_accuracy_decays_exp_step.svg">

![legend_accuracy_decays](https://user-images.githubusercontent.com/59259102/111536220-06d0b380-877b-11eb-8467-e2fdf29862b5.jpg)

* График функции потерь для нейронной сети EfficientNet-B0 (случайное начальное приближение):  


<img src="./epoch_loss_decays_exp_step.svg">


![legend_loss_decays](https://user-images.githubusercontent.com/59259102/111536241-0d5f2b00-877b-11eb-8afd-c5455de0508d.jpg)

* Анализ полученных результатов  


