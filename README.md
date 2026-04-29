# age_gender

#link model : https://drive.google.com/file/d/1kkPaW5veTiHn7RtLyowhfpL9zvKvkJ2g/view?usp=drive_link

Giới thiệu : Mô hình dự đoán tuổi và giới tính thông qua hình ảnh .

Công nghệ sử dụng:

Tensorflow, keras

Python

Mô hình : mô phỏng lại kết nối dư của Resnet , sử dụng các khối (block)


class ResidualBlock(keras.Model):
    
    def __init__(self, filters, stride=1):
        super().__init__()
        
        self.conv1 = layers.Conv2D(filters, (3,3),
                                   strides=stride,
                                   padding='same',
                                   use_bias=False)
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(filters, (3,3),
                                   padding='same',
                                   use_bias=False)
        self.bn2 = layers.BatchNormalization()

        self.shortcut_conv = layers.Conv2D(filters, (1,1),
                                           strides=stride,
                                           padding='same',
                                           use_bias=False)
        self.shortcut_bn = layers.BatchNormalization()

        self.stride = stride
        self.filters = filters

    def call(self, x, training=False):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        # projection shortcut
        if self.stride != 1 or shortcut.shape[-1] != self.filters:
            shortcut = self.shortcut_conv(shortcut)
            shortcut = self.shortcut_bn(shortcut, training=training)

        x = x + shortcut
        return tf.nn.relu(x)


class MyModel(keras.Model):
  
    def __init__(self):
        super().__init__()
        self.rescale = layers.Rescaling(1./255)
        self.conv1 = layers.Conv2D(32,(3,3),padding='same',use_bias=False)
        self.bn1 = layers.BatchNormalization()

        self.res1 = ResidualBlock(32)
        self.res2 = ResidualBlock(32)

        self.res3 = ResidualBlock(64, stride=2)
        self.res4 = ResidualBlock(64)

        self.res5 = ResidualBlock(128, stride=2)
        self.res6 = ResidualBlock(128)

        self.res7 = ResidualBlock(256, stride=2)
        self.res8 = ResidualBlock(256)

        self.res9 = ResidualBlock(512, stride=2)
        self.res10 = ResidualBlock(512)
        
        self.gap = layers.GlobalAveragePooling2D()  
        self.drop = layers.Dropout(0.5)
        self.age_head = layers.Dense(1, activation='linear', name='age')
        self.gender_head = layers.Dense(1, activation='sigmoid', name='gender')

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.res1(x, training=training)
        x = self.res2(x, training=training)

        x = self.res3(x, training=training)
        x = self.res4(x, training=training)

        x = self.res5(x, training=training)
        x = self.res6(x, training=training)

        x = self.res7(x, training=training)
        x = self.res8(x, training=training)

        x = self.res9(x, training=training)
        x = self.res10(x, training=training)

        x = self.gap(x)
        x = self.drop(x, training=training)
        age = self.age_head(x)
        gender = self.gender_head(x)
        
        return  {
    'age': age,
    'gender': gender}

Kết quả :

1. ĐÁNH GIÁ TỔNG QUAN TỪ KERAS

- loss: 1.1420
- compile_metrics: 76.5896
- age_loss: 0.3761
- gender_loss: 6.3894


 2. BÁO CÁO PHÂN LOẠI CHI TIẾT - GIỚI TÍNH

              precision    recall  f1-score   support

     Nam (0)       0.86      0.94      0.89       411
      Nữ (1)       0.93      0.83      0.88       389

    accuracy                           0.89       800
   macro avg       0.89      0.88      0.89       800
weighted avg       0.89      0.89      0.89       800


 3. MA TRẬN NHẦM LẪN (CONFUSION MATRIX)

                  | Đoán: Nam (0)   | Đoán: Nữ (1)   
-------------------------------------------------------
Thực tế: Nam (0)  | 385             | 26             
Thực tế: Nữ (1)   | 65              | 324            

