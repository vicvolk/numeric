
# coding: utf-8

###### Лабораторная по численным методам

""" 
Подключаем необходимые библиотеки
"""
import numpy as np 
# собственные числа можем получить стандартной функцией np.linalg.eig( A )
# from scipy.linalg import hessenberg - для сравнения с алгоритмом Хессенберга hessenberg( A ) 


# In[2]:

def dotproduct( x, y ): 
    """    
    Функция скалярного произведения векторов    
    """
    res = 0
    i = 0
    if( len(x[0]) != len(y[0]) ):
        print "Ошибка: векторы имеют разную длину!";
    else:
        for i in range( len(x[0]) ):
            res += x[0, i] * y[0, i]
    return res


# In[3]:

def matrixmultip( A, B ):
    """
    Функция умножения матриц
    Внимание! Векторы передаются построчно
    s - сдвиг по диагонали матрицы, для умножения части матрицы
    """
    if( A.shape[1] == B.shape[0] ):
        C = np.zeros( ( A.shape[0], B.shape[1] ) )
        for i in range( A.shape[0] ):
            for j in range( B.shape[1] ):
                el = 0
                for k in range ( B.shape[0] ):
                    el += A[i, k] * B[k, j]
                C[i, j] = el
        return C
    else :
        print "Ошибка: неподходящие размеры матриц!"
        
def vectormultip( xcol, xrow ):
    """
    Умножение вектора-столбца на вектор-строку
    """
    C = np.zeros( ( len(xcol[0]), len(xrow[0]) ) )
    for i in range( len(xcol[0]) ):
        for j in range( len(xrow[0]) ):
            C[i, j] = xcol[0, i] * xrow[0, j]
    return C

def matvecmultip( A, xcol ):
    """
    Умножением матрицы на вектор-стобец
    """
    C = np.zeros( ( A.shape[0], 1 ) )
    for i in range( A.shape[0] ):
        el = 0
        for j in range( A.shape[1] ):
            el += A[i, j] * xcol[j]
        C[i, 0] = el
    return C


# In[4]:

def matrixtranspose( A ):
    """
    Функция транспонирования матрицы
    """
    C = np.zeros( ( A.shape[1], A.shape[0] ) )
    for i in range( A.shape[0] ):
        for j in range( A.shape[1] ):
            C[j, i] = A[i, j]
    return C


# In[5]:

def getcol( A , j, s = 0):
    """
    Получение столбца матрицы
    """
    if(j >= 0 and j < A.shape[1] and s < A.shape[0]):
        res = np.zeros( A.shape[0] - s )
        k = 0
        for i in range ( s, A.shape[0] ):
            res[k] = A[i, j]
            k += 1
        return res
    else :
        print "Ошибка: выход за границы вектора!"


# In[6]:

def norm2( x ):
    """
    Евклидова норма вектора
    """
    res = 0
    for a in x:
        res += a * a
    return np.sqrt( res )


# In[7]:

def givens( a, b, i, sz):
    """
    Матрица Гивенса для двух элементов
    Литература: Jim Lambers, Lecture 9 Notes, The QR factorization, url: www.math.usm.edu/lambers/mat610/sum10/lecture9.pdf
    """
    if(abs(b) >= abs(a)):
        tau = a / b
        s = 1 / np.sqrt( 1 + tau * tau )
        c = tau * s
    else:
        tau = b / a
        c = 1 / np.sqrt( 1 + tau * tau )
        s = tau * c
    G = np.eye( sz, k = 0 )
    G[i - 1, i - 1] = c
    G[i - 1, i] = s
    G[i, i - 1] = -s
    G[i, i] = c
    return G

def givensQR( A ):
    """
    QR-разложение с помощью матриц вращения Гивенса
    """
    R = A
    Q = np.eye( max( A.shape[0], A.shape[1] ), k = 0 )
    vec = np.zeros(2)
    for j in range( A.shape[1] ):
        for i in range( A.shape[0] - 1, j, -1 ):
            G = givens( R[i - 1, j], R[i, j], i, max( A.shape[0], A.shape[1] ) )
            R = matrixmultip( G, R )           
            Q = matrixmultip( Q, matrixtranspose(G) )
    return [Q, R]
            


# In[8]:

def hous( x , sz ):
    """
    Вычисление вектора Хаусхолдера
    Литература: 
    1. Голуб, Ван Лоун. Алгоритм с нормировкой вектора по первому элементу
    2. Jim Lambers, Ibidem. Алгоритм без нормировки
    Делаем без нормировки, чтоб исключить возможность потери точности
    UPDATE: вычисляем сразу матрицу Хаусхолдера
    shift - для построения формы Хессенберга
    """
    n = len( x )
    v = x
    norm = norm2( x )
    if (norm != 0):
        v = x + np.sign(x[0]) * norm * np.eye( 1, n )         
    C = np.eye( sz, k = 0 )
    D = np.eye( len(v[0]), k = 0 ) - ( 2 / dotproduct(v, v) * vectormultip( v, v ) )
    if( n < sz ):
        ii = 0
        for i in range( sz - n, sz ):
            jj = 0
            for j in range(sz - n, sz ):
                C[i, j] = D[ii, jj]
                jj += 1
            ii += 1
    else:
        C = D
    return C

def housQR( A ):
    """
    QR - разложение методом Хаусхолдера
    """
    R = A
    Q = np.eye( max( A.shape[0], A.shape[1] ), k = 0 )
    for j in range( A.shape[1] ):
        vec = getcol( R, j, j)
        H = hous ( vec , A.shape[0] )
        R = matrixmultip( H, R )
        Q = matrixmultip( Q, H )
    return [Q, R]


# In[9]:

def hess( A ):
    """
    Приведение матрицы к форме Хесcенберга
    """
    R = A
    for j in range( A.shape[1] - 1):
        vec = getcol( R, j, j + 1 )
        if ( len(vec) == 1 ):
            break
        H = hous ( vec , A.shape[0] )
        R = matrixmultip( matrixmultip( H, R ) , matrixtranspose( H ))
    return R

def raylei( A , s = 0 ):
    """
    QR - шаг со сдвигом Рэлея
    матрица A должна быть квадратной
    """
    sz = A.shape[0]
    mu = A[sz - s - 1, sz - s - 1]
    A_k = A[0:sz - s, 0:sz - s] - mu * np.eye( sz - s, k = 0 ) # сдвиг
    [Q, R] = givensQR( A_k ) # разложение
    A[0:sz - s,0:sz - s] = matrixmultip( R, Q ) + mu * np.eye( sz - s, k = 0 ) # восстановление
    return A 

def wilk( A, s = 0 ):
    """
    QR - шаг со сдвигом Уилкинсона
    Литература: Уоткинс, Основы матричных вычислений - '... находим собственное значение наиболее близкое к a_nn ...'
    """
    sz = A.shape[0]
    T = A[sz - s - 2, sz - s - 2] + A[sz - s - 1, sz - s - 1] # След матрицы
    D = A[sz - s - 2, sz - s - 2] * A[sz - s - 1, sz - s - 1] - A[sz - s - 2, sz - s - 1]*A[sz - s - 1, sz - s - 2] # Определитель
    a0 = A[sz - s - 1, sz - s - 1]
    a1 = T / 2 + np.sqrt( T * T / 4 - D + 0j)
    a2 = T / 2 - np.sqrt( T * T / 4 - D + 0j)
    mu1 = np.sqrt( a1.real * a1.real + a1.imag * a1.imag )
    mu2 = np.sqrt( a2.real * a2.real + a2.imag * a2.imag )
    if( mu1 == mu2 ):
        mu = mu1
    elif ( abs(a0 - mu1) <= abs(a0 - mu2) ):
        mu = mu1
    else:
        mu = mu2        
    A_k = A[0:sz - s, 0:sz - s] - mu * np.eye( sz - s, k = 0 ) # сдвиг
    [Q, R] = givensQR( A_k ) # разложение
    A[0:sz - s,0:sz - s] = matrixmultip( R, Q ) + mu * np.eye( sz - s, k = 0 ) # восстановление
    return A

def wilk2( A, s ):
    """
    QR - шаг со сдвигом Уилкинсона, комплексный случай
    """
    sz = A.shape[0]
    T = A[sz - s - 2, sz - s - 2] + A[sz - s - 1, sz - s - 1] # След матрицы
    D = A[sz - s - 2, sz - s - 2] * A[sz - s - 1, sz - s - 1] - A[sz - s - 2, sz - s - 1]*A[sz - s - 1, sz - s - 2] # Определитель
    
    a0 = A[sz - s - 1, sz - s - 1]
    
    a1 = T / 2 + np.sqrt( T * T / 4 - D + 0j)
    a2 = T / 2 - np.sqrt( T * T / 4 - D + 0j)
    
    A_k = A[0:sz - s, 0:sz - s] - a1 * np.eye( sz - s, k = 0 ) # сдвиг 1
    [Q, R] = givensQR( A_k ) # разложение
    A[0:sz - s,0:sz - s] = matrixmultip( R, Q ) + a1 * np.eye( sz - s, k = 0 ) # восстановление
        
    A_k = A[0:sz - s, 0:sz - s] - a2 * np.eye( sz - s, k = 0 ) # сдвиг 2
    [Q, R] = givensQR( A_k ) # разложение
    A[0:sz - s,0:sz - s] = matrixmultip( R, Q ) + a2 * np.eye( sz - s, k = 0 ) # восстановление
    
    return A


# In[13]:

def QR( A , tp = 1 ):
    """
    QR - алгоритм поиска собственных значений
    """
    k = 40
    A_k = A
    delta = 1.0e-4
    if( tp == 3 or tp == 4 or tp == 5): # Если решаем с помощью сдвига Рэлея или Уилкинсона, то сначала приводим к форме Хессенберга
        A_k = hess( A )
        sz = A_k.shape[0]
        s = 0        
        # Инициализируем ведущий элемент
        el = A_k[sz - s - 1, sz - s - 1] 
        for i in range( k ):
            if( tp == 3 ): # Сдвиг Рэлея
                A_k = raylei( A_k , s ) 
            elif( tp == 4 ): # Один вещественный сдвиг Уилкинсона
                A_k = wilk( A_k , s )     
            elif( tp == 5 ): # Два сдвига Уилкинсона для случая комплексных корней
                A_k = wilk2( A_k , s ) 
            # Если ведущий элемент не сильно меняется при итерациях, то мы сошлись к собственному значению
            el_cur = A_k[sz - s - 1, sz - s - 1]            
            if ( abs(el - el_cur) < delta ):
                s += 1 # Сдвигаем ведущий столбец влево
                el = A_k[sz - s - 1, sz - s - 1] # Выставляем новый ведищий элемент
                if((sz - s) < 0):
                    break
                    
            else:
                el = el_cur
    else:     
        for i in range( k ):
            if( tp == 1 ): # Методом Гивенса
                [Q, R] = givensQR( A_k ) # Подходит и для неквадратных матриц
            elif( tp == 2 ): # Метод Хаусхолдера
                [Q, R] = housQR( A_k )
            A_k = matrixmultip( R, Q ) #! Только квадратные матрицы
    print A_k
