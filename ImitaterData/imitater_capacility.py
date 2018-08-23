import os
import datetime
import time
import random
import numpy as np
import sqlite3

from ImitaterData.imitater_user import *
from DQN_net import dqn

np.set_printoptions(suppress=True)

class Capacility_Create():
    def __init__(self):
        self.user_dict = {}
        self.capacility_state = np.ones(shape=[3])*1000
        self.capacility_state[0:2] = self.capacility_state[0:2]*1000

    def compute_decrease_capacility(self,actions):#"actions is a number"
        for i in range(len(self.capacility_state)):
            count = actions[actions==i]
            self.capacility_state[i]-=len(count)

    def compute_increase_capacility(self,actions):#"actions is a number"
        for i in range(len(self.capacility_state)):
            count = actions[actions==i]
            self.capacility_state[i]+=len(count)

if __name__ == '__main__':

    second = sleeptime(0, 0, 4);
    user_create = User_Create()

    i = 0
    while i < len(user_create.alluser_sequence):
        user_attributes = user_create.alluser_sequence[i]
        user_leavetime = user_attributes[-1]
        if op.ge(nowTime, user_leavetime):
            print('i', i)
            user_create.alluser_sequence.remove(user_attributes)
            del user_create.alluser_action[i]
            i -= 2
            print('i1', i)
        i += 1