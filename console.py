def pr(s,*argv,**argp):
    '''Print directly to stdout with no delay
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''
    print(s,*argv,**argp)
    sys.stdout.flush()


def progress(n,N,steps=100):
    '''Show progress - how many percent of total progress is made. n-current index and N-total count.
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''
        
    if (n % (N//steps)==0):
        print('{}'.format(100*n//N), end=' ')
        sys.stdout.flush()
    if (n==N-1):
        print('100% done.')
        sys.stdout.flush()
        
        



def get_str(X,precision=2):
    '''
    General string representation of lists,sets,tuples,np.ndarray,dicts,iterables with given precision for each floating point number.
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''
    
    sFormatString = '{{:.{}f}}'.format(precision)
    lsElements = [sFormatString.format(Element) for Element in X]
    if isinstance(X,tuple):
        sOpen = '('
        sClose = ')'
    elif isinstance(X,set):
        sOpen = 'set('
        sClose = ')'
    elif isinstance(X,list):
        sOpen = '['
        sClose = ']'
    elif isinstance(X,np.ndarray):
        sOpen = 'ndarray('
        sClose = ')'
    elif isinstance(X,dict):
        sOpen = '{'
        sClose = '}'
    else:
        sOpen = 'iterable('
        sClose = ')'

    sMiddle = ','.join(lsElements) + ','
    sMiddle=sMiddle.replace('.'+'0'*precision+',',',')  # replace all non-informative zeros

    sResult = sOpen+sMiddle[:-1]+sClose
    return sResult


def print_time(prefix='Execution time: '):
    '''Prints current time. 
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''
    
    print(time.strftime(prefix+"%Y-%m-%d %H:%M:%S", time.localtime()))        