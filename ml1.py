import numpy as np
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os 
from PIL import Image
cat = []
ycat = []
dog = []
ydog = []

def convert(imagename) :
        img = Image.open(imagename) # image extension *.png,*.jpg

        new_width  = 50
        new_height = 50
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        img.save(imagename) 
        data = np.asarray(img, dtype="int32" )
        def rgb2gray(rgb):
            return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        img = mpimg.imread(imagename)     
        gray = rgb2gray(img) 
        data = np.asarray(gray, dtype="int32" )
        data =  data.flatten('F')
        
        return data
 # curr_dir = os.path.join(os.path.sep, fruit_dir)
all_imgs = os.listdir(os.getcwd()+ "/cats")
for img_file in all_imgs:
    cat.append(convert(os.getcwd()+"/cats"+'/'+img_file))
    ycat.append(1)
print(len(ycat))    
all_imgs = os.listdir(os.getcwd()+ "/dogs")
for img_file in all_imgs:
    dog.append(convert(os.getcwd()+"/dogs"+'/'+img_file))
    ydog.append(0)


    
X = []
cat =  np.array(cat)
dog =  np.array(dog)
y = [[]]
ycat =  np.array(ycat)
ydog =  np.array(ydog)


X = np.concatenate((dog,cat))
y = np.concatenate((ydog,ycat))

y = np.reshape(y,(-1,1))



from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=0)
   
x_train = X[0:20000, :]
y_train = y[0:20000,:]
x_test = X[20000:, :]
y_test = y[20000:,:]

X, y = shuffle(x_train, y_train, random_state=0)
x_test, y_test = shuffle(x_test, y_test, random_state=0)
print(y_test)
# Each row is a training example, each column is a feature  [X1, X2, X3]
# X=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
# y=np.array(([0],[1],[1],[0]), dtype=float)

# Define useful functions    

# Activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)

# Class definition
class NeuralNetwork:
    def __init__(self, x,y):
        self.input = x
        self.weights1= np.random.rand(self.input.shape[1],10) # considering we have 4 nodes in the hidden layer
        self.weights2 = np.random.rand(10,10)
        self.weights3 = np.random.rand(10,10)
        self.y = y
        self.output = np.zeros(y.shape)
        
    def feedforward(self):
        # print(self.input.shape[1])
        # print(self.weights1.shape)
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.layer3 = sigmoid(np.dot(self.layer2, self.weights3))
        # print(self.layer1.shape)
        # print(self.layer2.shape)
        return self.layer2
        
    def backprop(self):
          
        d_weights3 = np.dot(self.layer2.T, 2*(self.y -self.output)*sigmoid_derivative(self.output))
        k = 2*(self.y -self.output)*sigmoid_derivative(self.output)
        d_weights2 = np.dot(self.layer1.T, np.dot(k, self.weights3.T)*sigmoid_derivative(self.layer2))
        k = np.dot(k, self.weights3.T)*sigmoid_derivative(self.layer2)
        d_weights1 = np.dot(self.input.T, np.dot(k , self.weights2.T)*sigmoid_derivative(self.layer1))
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.weights3 += d_weights3

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()
    def test(self):
        self.input = x_test
        self.y = y_test
        self.feedforward()


NN = NeuralNetwork(X,y)
for i in range(10): # trains the NN 1,500 times
    print(i)
    if i % 10 ==0: 
        print ("for iteration # " + str(i) + "\n")
        print ("Input : \n" + str(X))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(NN.feedforward()))
        print ("Loss: \n" + str(np.mean(np.square(y - NN.feedforward())))) # mean sum squared loss
        print ("\n")
  
    NN.train(X, y)   
print("********")    
print ("Input : \n" + str(x_test))
print ("Actual Output: \n" + str(y_test))
NN.test()
print("********") 
print(NN.feedforward())
out = NN.feedforward()

cat_cat  = 0
cat_dog  = 0
dog_cat  = 0
dog_dog  = 0

for i in range(1,4999):
    if y_test[i][0] ==1 and int(out[i][0]) == 1:
       
       cat_cat = cat_cat +1
    if y_test[i][0] ==1 and int(out[i][0]) == 0:
       
       cat_dog = cat_dog +1
    if y_test[i][0] ==0 and int(out[i][0]) == 1:
       
       dog_cat  =dog_cat +1
    if y_test[i][0] ==0 and int(out[i][0]) == 0:
       
       dog_dog = dog_dog +1   
print("cat_cat") 
print(cat_cat )

print("cat_dog ")      

print(cat_dog )
print("dog_cat ") 
print(dog_cat )
print("dog_dog ")   
print(dog_dog )
print ("Loss: \n" + str(np.mean(np.square(y_test - NN.feedforward())))) # mean sum squared loss
print ("\n")
