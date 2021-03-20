# Исследование влияния параметра “темп обучения” на процесс обучения нейронной сети на примере решения задачи классификации Oregon Wildlife с использованием техники обучения Transfer Learning  
***  
### Обучение нейронной сети EfficientNet-B0 (предварительно обученной на базе изображений imagenet) для решения задачи классификации изображений Oregon WildLife с использованием фиксированных темпов обучения 0.1, 0.01, 0.001, 0.0001
В данной лабораторной работе для решения задачи классификации изображений Oregon Wildlife использовалась нейронная сеть EfficientNet-B0, причем данная нейронная сеть будет иметь уже предобученные веса на базе изображений ImageNet (weights = "imagenet"). Также убирается классификатор данной нейронной сети с помощью параметра include_top=False, и создается собственный классификатор (один слой Flatten и полносвязный Dense слой). С помощью model.trainable = False мы замораживаем ту часть нейронной сети, которая отвечает за выделение каких-то характерных признаков в изображении, так как эта часть уже обучена:  
```
 inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))  
  model = EfficientNetB0(include_top=False, input_tensor=inputs,pooling = 'avg', weights='imagenet')  
  model.trainable = False  
  model = tf.keras.layers.Flatten()(model.output)  
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.softmax)(model)  
  return tf.keras.Model(inputs=inputs, outputs=outputs)  
  ```
В процессе лабораторной работы изменялся темп обучения (0.1, 0.01, 0.001, 0.0001). Данное изменение позволит выявить более оптимальный темп обучения для решения задачи классификации изображений Oregon Wildlife, исходя из графиков метрики точности, графиков функции потерь и скорости обучения:
```
#optimizer=tf.optimizers.Adam(lr=0.1)
#optimizer=tf.optimizers.Adam(lr=0.01)
#optimizer=tf.optimizers.Adam(lr=0.001)
#optimizer=tf.optimizers.Adam(lr=0.0001)
```
### В результате обучения нейронной сети EfficientNet-B0 (предобученной) с разными темпами обучения (0.1 0.01 0.001 0.0001) получили следующие графики:    
* График метрики точности для предобученной нейронной сети EfficientNet-B0 с различными темпами обучения (0.1 0.01 0.001 0.0001):  

![legend_accuracy](https://user-images.githubusercontent.com/59259102/111492129-77f97200-874d-11eb-963e-ec33367dd4a5.jpg) 

<img src="./epoch_categorical_accuracy_for_different_lr.svg">

* График функции потерь для предобученной нейронной сети EfficientNet-B0 с различными темпами обучения (0.1 0.01 0.001 0.0001):  

![legend_loss](https://user-images.githubusercontent.com/59259102/111492201-86e02480-874d-11eb-8213-10a1a49438bb.jpg)

<img src="./epoch_loss_for_different_lr_1.svg">

<img src="./epoch_loss_for_different_lr.svg">

* Анализ полученных результатов  
  + Исходя из полученных результатов можно отметить следующее: на тренировочном наборе данных лучшей метрики точности удалось добиться с темпом обучения равным 0.001, так как на протяжении всего обучения значения точности были выше, чем у остальных, и в конце обучения также наблюдается превосходство данного темпа обучения над остальными (точность около 94 процентов). Что же касается метрики точности валидационнго набора данных, то наблюдается следующее: с одной стороны лучший результат в конце обучения получился с использованием темпа обучения 0.1 (точность 88.5 процентов), но можно заметить, что при таком темпе обучения график получился "изрезанным", а это значит, что на протяжении всего обучения с каждой эпохой значения метрики точности сильно отличаются друг от друга и резко меняются. Похожая ситуация и с темпом равным 0.01. Поэтому можно отметить темпы обучения равные 0.0001 и 0.001, так как у них на протяжении всего обучения не наблюдается резкого изменения значения метрики точности, и в конце точность (для темпа 0.0001 точность 87.79 процентов, а для темпа 0.001 точность 88 процентов) получилось схожая с темпами обучения 0.01 (точность 87.69 процентов) и 0.1 (точность 88.5 процентов).  
  + Что касается функции потерь, то можно сказать, что лучший результат получился с темпами обучения равными 0.0001 и 0.001, так как на протяжении почти всего обучения потери были ниже чем у остальных на валидационном наборе данных и на тренировочном. Худшим же оказался темп обучения равный 0.1, так как с каждой эпохой на протяжении всего обучения величина ошибки возрастает на валидационном наборе данных, а на тренировочном наблюдаются значения потерь выше чем у других темпов обучения.  
  + Исходя из всего вышесказанного, можно отметить темпы обучения 0.001 и 0.0001 как самые оптимальные из всех использовавшихся. Однако, обратив внимание на скорость обучения, можно сказать что темп обучения 0.001 самый оптимальный , так как у него скорость обучения выше (12 минут 50 секунд), чем у темпа обучения 0.0001 (25 минут 29 секунд). Отличие почти в 2 раза.   
***  
## Реализация и применение в обучении следующих политик изменения темпа обучения: a. Пошаговое затухание (Step Decay) b. Экспоненциальное затухание (Exponential Decay)  
Политика пошагового затухания предполагает, что существует определенный начальный темп обучения и с наступлением определенного времени (эпохи) данный начальный темп будет уменьшаться в некоторое количество раз. Таким образом, темп обучения будет снижаться в несколько раз каждые несколько эпох.  
```
def step_decay(epoch,lr): # epoch - текущая эпоха  
  initial_lrate = 0.001 # начальный темп обучения  
  drop = 0.5 # во сколько раз будет изменяться тепм обучения (увеличиваться в 0.5 раз, т.е. уменьшаться в 2 раза)  
  epochs_drop = 5.0 # каждую пятую эпоху будет изменяться темп обучения  
  lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))  
  return lrate  
```
Политика Экспоненциального затухания предполагает изменение начального темпа обучения в ```e``` в определенной степени раз каждые несколько эпох. Таким образом, темп обучения будет снижаться примерно в 2,7 в определенной степени раз каждую эпоху.    
```
def exp_decay(epoch,lr): # epoch - текущая эпоха  
  initial_lrate = 0.001 # начальный темп обучения  
  k = 0.11 # коэффициент в степени числа Эйлера  
  lrate = initial_lrate * math.exp(-k*epoch)  
  return lrate  
```
## Нахождение оптимальных параметров для экспоненциального затухания
Для нахождения оптимальных параметров экспоненциального затухания, были проведены обучения с различными параметрами, такими как (initial_lrate - начальный темп обучения, k - коэффициент наклона экспоненциальной кривой):  
+ initial_lrate = 0.1 и k = 0.1  
+ initial_lrate = 0.1 и k = 0.2  
+ initial_lrate = 0.1 и k = 0.3  
+ initial_lrate = 0.01 и k = 0.1  
+ initial_lrate = 0.01 и k = 0.2  
+ initial_lrate = 0.01 и k = 0.3  
+ initial_lrate = 0.001 и k = 0.1  
В результате получили следующие графики:  
* График метрики точности для предобученной нейронной сети EfficientNet-B0 (экспоненциальное затухание) для валидационного набора данных:   

![image](https://user-images.githubusercontent.com/59259102/111868798-b167f280-8984-11eb-8110-39d823760b21.png)  


<img src="./epoch_categorical_accuracy_exp_diff_param.svg">  

* График функции потерь для предобученной нейронной сети EfficientNet-B0 (экспоненциальное затухание) для валидационного набора данных:    

![image](https://user-images.githubusercontent.com/59259102/111868873-20dde200-8985-11eb-89bd-c84f2d3f365f.png)  


<img src="./epoch_loss_exp_diff_param.svg">   

<img src="./epoch_loss_exp_diff_param1.svg">   


***
## Нахождение оптимальных параметров для пошагового затухания  
Для нахождения оптимальных параметров пошагового затухания, были проведены обучения с различными параметрами, такими как (initial_lrate - начальный темп обучения, drop - во сколько раз будет изменяться тепм обучения, epochs_drop - эпоха на которой будет изменяться темп обучения):  
+ initial_lrate = 0.1, drop = 0.5, epochs_drop = 10.0  
+ initial_lrate = 0.1, drop = 0.5, epochs_drop = 5.0 
+ initial_lrate = 0.1, drop = 0.5, epochs_drop = 3.0  
+ initial_lrate = 0.01, drop = 0.5, epochs_drop = 10.0  
+ initial_lrate = 0.01, drop = 0.5, epochs_drop = 5.0
+ initial_lrate = 0.01, drop = 0.5, epochs_drop = 3.0 
+ initial_lrate = 0.001, drop = 0.5, epochs_drop = 5.0  
В результате получили следующие графики:  
* График метрики точности для предобученной нейронной сети EfficientNet-B0 (экспоненциальное затухание):  



* График функции потерь для предобученной нейронной сети EfficientNet-B0 (экспоненциальное затухание):  


***
### В результате обучения нейронной сети EfficientNet-B0 (предобученной) с применением политик изменения темпа обучения (пошаговое затухание и экспоненциальное затухание) получили следующие графики:  
 
* График метрики точности для предобученной нейронной сети EfficientNet-B0 (пошаговое затухание и экспоненциальное затухание):  
 
![legend_accuracy_decays](https://user-images.githubusercontent.com/59259102/111536220-06d0b380-877b-11eb-8467-e2fdf29862b5.jpg)  

<img src="./epoch_categorical_accuracy_decays_exp_step.svg">  

* График функции потерь для предобученной нейронной сети EfficientNet-B0 (пошаговое затухание и экспоненциальное затухание):  

![legend_loss_decays](https://user-images.githubusercontent.com/59259102/111536241-0d5f2b00-877b-11eb-8afd-c5455de0508d.jpg)  

<img src="./epoch_loss_decays_exp_step.svg">  

* Анализ полученных результатов  
Исходя из полученных результатов можно отметить следующее:  
  + В случае с пошаговым затуханием удалось превзойти результаты оптимально выбранного фиксированного темпа обучения 0.001, а именно: увеличилась скорость обучения (11 минут 48 секунд), увеличилась метрика точности на валидационном наборе данных в конце обучения (88.89 процентов у пошагового затухания, 88.03 процента у фиксированного темпа 0.001), уменьшилось значение функции потерь на валидационном наборе данных в конце обучения (0.2216 у пошагового затухания, 0.3691 у фиксированного темпа 0.001). Таким образом подобранные параметры (initial_lrate = 0.001, drop = 0.5, epochs_drop = 5.0) для пошагового затухания являются оптимальными, так как это привело к улучшению сходимости алгоритма обучения и по скорости, и по точности, и потерь также меньше в сравнении с оптимально выбранным фиксированным темпом обучения 0.001. Но все же нельзя сказать, что данные параметры самые оптимальные и существуют ли более оптимальные параметры для пошагового затухания (для данных параметров метрика качества не намного улучшилась, около 0.8 процента).  
  + Почти аналогичную ситуацию можно наблюдать при использовании экспоненциального затухания: увеличилась метрика точности на валидационном наборе данных в конце обучения (88.6 процентов у экспоненциального затухания, 88.03 процента у фиксированного темпа 0.001), уменьшилось значение функции потерь на валидационном наборе данных в конце обучения (0.2386 у экспоненциального затухания, 0.3691 у фиксированного темпа 0.001). Однако скорость обучения у экспоненциального затухания уменьшилась (15 минут 50 секунд) по сравнению с фиксированным темпом 0.001 (12 минут 50 секунд). Это означает, что параметры (initial_lrate = 0.001 и k = 0.11) для экспоненциального затухания возможно не оптимальные. Для этого вносились изменения в параметры экспоненциального затухания: 
    + При initial_lrate = 0.1 и k = 0.11 скорость обучения (11 минут 41 секунды) была выше, точность (87.95) ниже и потери (2.84) больше по сравнению с темпом 0.001 на валидации. 
    + При initial_lrate = 0.01 и k = 0.11 также скорость обучения (11 минут 43 секунды) была выше, точность (87.74) ниже и потери (0.4334) больше по сравнению с темпом 0.001 на валидации. 
    + При initial_lrate = 0.1 и k = 0.2 скорость обучения (19 минут 54 секунды) была ниже, точность (88.73) выше и потери (1.47) больше по сравнению с темпом 0.001 на валидации.   
Поэтому у экспоненциального затухания не было найдено тех оптимальных параметров, которые давали бы и лучше точность, и меньше потерь, и при этом скорость обучения была бы выше по сравнению с фиксированным темпом 0.001 на валидации. Поэтому из всех перечисленных параметров были выбраны те, которые дают прирост по двум (точность, потери) из трех возможных пунктов (точность, потери, скорость обучения) - initial_lrate = 0.001 и k = 0.11.

