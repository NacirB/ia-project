import numpy as np
import math
from joblib.numpy_pickle_utils import xrange
from sklearn import preprocessing
import pyodbc

#get data from sql server

def getData1(conn):
    cursor = conn.cursor()
    cursor.execute("select * from president_T1 where [Libellé du département] = 'Cher' order by  [Libellé de la commune]")
    data1 = cursor.fetchall()
    return data1

def getData2(conn):
    cursor = conn.cursor()
    cursor.execute("select * from president_T2 where [Libellé du département] = 'Cher' order  by [Libellé de la commune]")
    data2 = cursor.fetchall()
    return data2



conn = pyodbc.connect(
    "Driver={SQL Server Native Client 11.0};"
    "Server=XPSTRIET;"
    "Database=Election;"
    "Trusted_Connection=yes;")

data1 = getData1(conn)
data2 = getData2(conn)
conn.close()


pred_data_T1 = list()
for countRow in range(0, len(data1)):

    row = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    i=0
    while (i < 33):
        if (data1[countRow][i] == 'MACRON' and data1[countRow][i + 1] == 'Emmanuel'):
            row[0] = data1[countRow][i + 2]
        elif (data1[countRow][i] == 'LE PEN' and data1[countRow][i + 1] == 'Marine'):
            row[1] = data1[countRow][i + 2]
        elif (data1[countRow][i] == 'FILLON' and data1[countRow][i + 1] == 'François'):
            row[2] = data1[countRow][i + 2]
        elif (data1[countRow][i] == 'MÉLENCHON' and data1[countRow][i + 1] == 'Jean-Luc'):
            row[3] = data1[countRow][i + 2]
        elif (data1[countRow][i] == 'HAMON' and data1[countRow][i + 1] == 'Benoît'):
            row[4] = data1[countRow][i + 2]
        elif (data1[countRow][i] == 'DUPONT-AIGNAN' and data1[countRow][i + 1] == 'Nicolas'):
            row[5] = data1[countRow][i + 2]
        elif (data1[countRow][i] == 'LASSALLE' and data1[countRow][i + 1] == 'Jean'):
            row[6] = data1[countRow][i + 2]
        elif (data1[countRow][i] == 'POUTOU' and data1[countRow][i + 1] == 'Philippe'):
            row[7] = data1[countRow][i + 2]
        elif (data1[countRow][i] == 'ASSELINEAU' and data1[countRow][i + 1] == 'François'):
            row[8] = data1[countRow][i + 2]
        elif (data1[countRow][i] == 'ARTHAUD' and data1[countRow][i + 1] == 'Nathalie'):
            row[9] = data1[countRow][i + 2]
        elif (data1[countRow][i] == 'CHEMINADE' and data1[countRow][i + 1] == 'Jacques'):
            row[10] = data1[countRow][i + 2]
        i = i + 3
    k=11
    for i in range(33, 36):
        row[k] = data1[countRow][i]
        k = k + 1;
    pred_data_T1.append(row)

#at this point, we have pred_data_T1



pred_data_T2 = list()
for countRow in range(0, len(data2)):

    row = [0, 0, 0, 0, 0]
    i = 0
    while (i < 6):
        if (data2[countRow][i] == 'MACRON' and data2[countRow][i + 1] == 'Emmanuel'):
            row[0] = data2[countRow][i + 2]
        elif (data2[countRow][i] == 'LE PEN' and data2[countRow][i + 1] == 'Marine'):
            row[1] = data2[countRow][i + 2]
        i = i + 3
    k = 2
    for i in range(6,9):
        row[k] = data2[countRow][i]
        k = k + 1
    pred_data_T2.append(row)

#at this point, we have pred_data_T2

test1 = list()
for i in range(0,100):
    test1.append(pred_data_T1[i])
test2 = list()
for i in range(0,100):
    test2.append(pred_data_T2[i])

syn1 = []
min_error =10000

def all_process(x,w,y,nb_of_iterations,learning_rate):
    global min_error
    global syn1
    for iter in xrange(nb_of_iterations):
        # forward propagation
        l0 = x
        l1 = nn(l0, w)
        # dimension l1 : (1,5)
        print("output ********************************\n", l1)
        print("***************************************")
        # how much did we miss?
        l1_error = loss(l1, y)

        print("average iteration error *************************************** \n", l1_error)
        print("***************************************")

        # multiply how much we missed by the
        # slope of the sigmoid at the values in l1
        # l1_delta = l1_error * nonlin(l1,True)
        if (l1_error < min_error):
            print("found a minimum local error \n")
            min_error = l1_error
            syn1 = syn0

        l1_delta = delta_w(w, l0, y, learning_rate)
        # dimension l1_delta (1,5)
        print("delta *************************************** \n", l1_delta)
        print("***************************************")
        print("****************************************MATRIX READJUSTMENT \n")
        w += np.dot(l0.T, l1_delta)


        # update weights
        # syn0 += np.dot(l0.T,l1_delta)




# sigmoid function

def nn(x, w):
    """Output function y = x * w"""
    return np.dot(x, w)


def loss(y, t):
    """MSE loss function"""
    return np.mean((t - y) ** 2)


def gradient(w, x, t):
    """Gradient function. (Remember that y = nn(x, w) = x * w)"""
    return 2 * np.dot(x.T, (nn(x, w) - t))


def delta_w(w_k, x, t, learning_rate):
    """Update function delta w"""
    return -learning_rate * np.mean(gradient(w_k, x, t))

# input dataset
learning_rate = 0.5
nb_of_iterations = 3


#  weights matrix
syn0 = np.array([[1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0],
                 [0.5, 0.2, 0.1, 0, 0.2],
                 [0.4, 0.1, 0.15, 0.05, 0.3],
                 [0.7, 0.05, 0.1, 0.05, 0.1],
                 [0.4, 0.3, 0.1, 0.1, 0.1],
                 [0.3, 0.3, 0.2, 0.1, 0.1],
                 [0.1, 0.3, 0.15, 0.15, 0.3],
                 [0.6, 0.2, 0.075, 0.025, 0.1],
                 [0.2, 0.4, 0.15, 0.15, 0.1],
                 [0.7, 0, 0.2, 0.05, 0.05],
                 [0.5, 0.1, 0.4, 0, 0],
                 [0.3, 0.1, 0, 0.1, 0.5],
                 [0.25, 0, 0.025, 0.025, 0.7]])
print( "weights *************************************** \n", syn0)

print("***************************************")

for k in range(0,len(test1)):
    X = np.array(test1[k])

    print("input : ********************************\n", X)

    print("***************************************")
    X = preprocessing.normalize([X])

    print("Trasformed Input : ********************************\n", X)

    print("***************************************")
    # output expected
    y = np.array(test2[k])

    y = preprocessing.normalize([y])

    print("transformed Expected Output *************************************** \n", y)
    print("***************************************")

    all_process(X,syn0,y,nb_of_iterations,learning_rate)

print("minimum average error : ",min_error)
print ("Matrix of weights After Training:")
syn1[syn1 < 0] = 0
print (syn1)




#Mtest = [[test1],[test2],[test3]]
#Mexpected = [[expected1],[expected2],[expected3]]

