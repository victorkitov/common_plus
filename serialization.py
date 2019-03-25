#!/usr/bin/env python
# encoding: utf-8
# Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016


import pickle
import shelve
import types


def save_all_variables(env,filename):
    '''Save all variables to file.
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''

    my_shelf = shelve.open(filename,'n') # 'n' for new
    #print env
    for key,val in env.items():
        if not key.startswith('__'):
            if not isinstance(val,(types.ModuleType,types.FunctionType,type)):
            #if not isinstance(val,(types.ModuleType,types.TypeType)):
                try:
                    #print 'trying:',key,type(key)
                    my_shelf[key] = env[key]
                except:
                    pass
                    #print('ERROR shelving: {0}'.format(key))
                else:
                    pass
                    #print 'OK shelving: {}={}'.format(key,env[key])
    my_shelf.close()



def load_all_variables(filename):
    '''
    Load all variables from file.
    
    Example:

    File1:
    a=1
    P=Struct(name='Andrew',age=23)
    save_all_variables(globals(),'delme.shl')

    File2:
    globals().update(load_all_variables('delme.shl'))
    print a
    print P
    
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''
    
    my_shelf = shelve.open(filename)
    env={}
    for key in my_shelf:
        env[key]=my_shelf[key]
    my_shelf.close()
    return env