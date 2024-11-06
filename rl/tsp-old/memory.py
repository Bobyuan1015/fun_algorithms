#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
REINFORCEMENT LEARNING

Started on the 25/08/2017

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""



from collections import deque




class Memory(object):
    def __init__(self,max_memory = 2000):
        self.cache = deque(maxlen=max_memory)
        # self.cache 是一个具有最大长度的双端队列（deque）。已达到 max_memory，那么新的元素会被添加进去，同时最旧的元素会被自动移除

    def save(self,args):
        self.cache.append(args)

    def empty_cache(self):
        self.__init__()



