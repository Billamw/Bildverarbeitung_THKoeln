def Aufg1():
    print("Hell Word")

def Aufg2(n):
    for x in range(n):
        print(x+1)

def Aufg3(n):
    if(n<1):
        print("richtige Zahl eingeben!")
    if(0<n):
        print(0)
    if(1<n):
        print(1)
    if(2<n):
        fibonacci = [None] * n
        fibonacci[0]=0
        fibonacci[1]=1

        for i in range(2, n):
            fibonacci[i]=fibonacci[i-1]+fibonacci[i-2]
            print(fibonacci[i])

def Aufg4(n):
    if(n<1):
        print("Zahl größer 0 wählen!")
    x=3;
    primenumbers = [2]

    while(len(primenumbers)<n):
        if(isPrimeNumber(x,primenumbers)==True):
            primenumbers.append(x)
        x=x+1

    print(primenumbers)

def isPrimeNumber(x, primenumbers):
    ret = True
    for i in range(len(primenumbers)):
        if((x%primenumbers[i])==0):
            ret = False
            break
    return ret;

Aufg4(6)