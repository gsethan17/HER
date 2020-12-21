from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, ReLU, MaxPool2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras import Model, Sequential
from src.ResidualBlock import ResidualBlock, ResidualBlock34

class ResNet(Model) :
    def __init__(self, num_layer = 50, se = None, n_classes = 2) : 
        super(ResNet, self).__init__()
        self.n_classes = n_classes
        self.num_layer = num_layer
        if self.num_layer == 50 :
            self.reps = [3, 4, 6, 3]
        elif self.num_layer == 101 : 
            self.reps = [3, 4, 23, 3]
        elif self.num_layer == 152 :
            self.reps = [3, 8, 36, 3]
            
        self.se = se
        self.conv_1 = Conv2D(filters = 64,
                             kernel_size = 7,
                             padding = 'same',
                             strides = 2,
                             kernel_initializer = 'he_normal',
                             name = 'ResNet_1')
        self.bn_1 = BatchNormalization(momentum = 0.9, name = 'ResNet_1_BN')
        self.relu_1 = ReLU(name = 'ResNet_1_Act')
        self.maxpool = MaxPool2D(3, 2,
                                    padding = 'same', 
                                    name = 'ResNet_1_Pool')
        self.residual_blocks = Sequential()
        for n_filters, reps in zip([[64, 64, 256], 
                                   [128, 128, 512], 
                                   [256, 256, 1024], 
                                   [512, 512, 2048]], 
                                   self.reps) :
            for i in range(reps) :
                if i == 0 :
                    if n_filters[0] == 64 :
                        self.residual_blocks.add(ResidualBlock(block_type = 'first_conv', 
                                                 n_filters = n_filters, se = self.se))
                    else : 
                        self.residual_blocks.add(ResidualBlock(block_type = 'conv',
                                                 n_filters = n_filters, se = self.se))
                else : 
                    self.residual_blocks.add(ResidualBlock(block_type = 'identity',
                                             n_filters = n_filters, se = self.se))
        self.GAP = GlobalAveragePooling2D()
        self.FC = Dense(units = self.n_classes, activation = 'tanh')


    def call(self, input, training = False) :
        x = self.conv_1(input)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.maxpool(x)
        x = self.residual_blocks(x)
        x = self.GAP(x)
        x = self.FC(x)

        return x


class ResNet34(Model) :
    def __init__(self, num_layer = 34, cardinality = None, se = None, adv = False, n_classes = 2) : 
        super(ResNet34, self).__init__()
        self.n_classes = n_classes
        self.num_layer = num_layer
        self.cardinality = cardinality
        self.se = se
        self.adv = adv
        self.conv_1 = Conv2D(filters = 64,
                             kernel_size = 7,
                             padding = 'same',
                             strides = 2,
                             kernel_initializer = 'he_normal',
                             name = 'ResNet_1')
        self.bn_1 = BatchNormalization(momentum = 0.9, name = 'ResNet_1_BN')
        self.relu_1 = ReLU(name = 'ResNet_1_Act')
        self.maxpool = MaxPool2D(3, 2,
                                    padding = 'same', 
                                    name = 'ResNet_1_Pool')
        self.residual_blocks = Sequential()
        if self.num_layer == 34 :
            self.zip = zip([64, 128, 256, 512], [3, 4, 6, 3], [False, True, True, True])
        elif self.num_layer == 18 :
            self.zip = zip([64, 128, 256, 512], [2, 2, 2, 2], [False, True, True, True])
        
        for n_filters, reps, downscale in self.zip :
            for i in range(reps) :
                if i == 0 and downscale :
                    self.residual_blocks.add(ResidualBlock34(block_type = 'conv', 
                                                 n_filters = n_filters, cardinality = self.cardinality, se = self.se))
                else : 
                    self.residual_blocks.add(ResidualBlock34(block_type = 'identity',
                                             n_filters = n_filters, cardinality = self.cardinality, se = self.se))
        self.GAP = GlobalAveragePooling2D()
        #####
        self.FC_1 = Dense(units = 512, activation = 'relu')
        self.DO_1 = Dropout(0.5)
        self.FC_2 = Dense(units = 512, activation = 'relu')
        self.DO_2 = Dropout(0.5)
        #####
        self.FC = Dense(units = self.n_classes, activation = 'tanh')

    def call(self, input) :
        x = self.conv_1(input)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.maxpool(x)
        x = self.residual_blocks(x)
        x = self.GAP(x)
        
        if self.adv :
            x = self.FC_1(x)
            x = self.DO_1(x)
            x = self.FC_2(x)
            x = self.DO_2(x)
        
        x = self.FC(x)

        return x
