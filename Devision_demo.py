from ImitaterData import imitater_user
from ImitaterData import imitater_capacility
from DQN_net import dqn
import numpy as np
import datetime
import operator as op
import random
import time
import copy
EPISDOE = 10000
STEP = 10000
LAMBDA = 0.01

class Devision():
    def __init__(self):
        self.user_create = imitater_user.User_Create()
        self.capacility_create = imitater_capacility.Capacility_Create()

    def generate_state(self):
        start_len = len(self.user_create.alluser_sequence)
        capacility = copy.deepcopy(self.capacility_create.capacility_state)
        if len(self.user_create.alluser_sequence) > 0:
            # print(len(self.user_create.alluser_sequence))
            nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            i = 0;
            while i < len(self.user_create.alluser_sequence):
                user_attributes = self.user_create.alluser_sequence[i]
                user_leavetime = user_attributes[-1]
                if op.ge(nowTime, user_leavetime):
                    self.user_create.alluser_sequence.remove(user_attributes)
                    actions = self.user_create.alluser_action[i]
                    del self.user_create.alluser_action[i]
                    self.capacility_create.compute_increase_capacility(actions)
                    i-=1
                i+=1
        end_len = len(self.user_create.alluser_sequence)
        print(end_len-start_len,self.capacility_create.capacility_state-capacility)
        self.user_create.Imitate_User()
        self.user_create.alluser_sequence.extend(self.user_create.currentuser_sequence)



    def relu(self,x):
        """Compute softmax values for each sets of scores in x."""
        x[x<0] = 0
        return x
    def train(self):
        action_dim = 3
        state_dim = action_dim * 2
        agent = dqn.DQN(action_dim, state_dim)
        while 1 == 1:

            for episode in range(EPISDOE):

                total_reward = 0
                # self.user_create = imitater_user.User_Create()

                self.generate_state()
                user_state_list = self.user_create.currentuser_sequence
                user_states = []

                for user_list in user_state_list:
                    user_states.append(user_list[1:4])

                print('start ')
                for step in range(STEP):
                    time.sleep(0)
                    capacility_state = copy.deepcopy(self.capacility_create.capacility_state)
                    capacility_states = np.tile(capacility_state, [len(user_states), 1])
                    state = np.hstack((user_states, capacility_states))
                    actions = agent.get_action(state)
                    user_states = np.array(user_states)
                    accept_propabilitys = np.zeros(shape=[len(user_states)])
                    for i in range(len(user_states)):
                        accept_propabilitys[i] = user_states[i,actions[i]]
                    rand_props = np.array([random.uniform(0,1) for _ in range(len(actions))])
                    g_at = rand_props-accept_propabilitys
                    g_at[g_at<0.0] = 0
                    # actions[g_at==0] = action_dim-1
                    g_at[g_at>0.0] = 1


                    #compute next_state
                    self.user_create.alluser_action.extend(actions)
                    self.generate_state()
                    next_user_state_list = self.user_create.currentuser_sequence
                    next_user_states = []
                    for next_user_list in next_user_state_list:
                        next_user_states.append(next_user_list[1:4])
                    self.capacility_create.compute_decrease_capacility(actions)
                    next_capacility_state = copy.deepcopy(self.capacility_create.capacility_state)
                    next_capacility_states = np.tile(next_capacility_state, [len(next_user_states), 1])
                    next_state = np.hstack((next_user_states, next_capacility_states))
                    # reward = g_at - LAMBDA * np.max(self.relu(-1 * next_capacility_states))
                    reward = copy.deepcopy(next_capacility_states)
                    print('reward',reward)
                    reward[reward[:,2]>0,0:2] = -np.abs(reward[reward[:,2]>0,0:2])//1000;
                    print('reward1',reward)
                    total_reward += np.sum(reward)
                    agent.percieve(state,actions,reward,next_state,False)
                    print('now_capacility', capacility_state)
                    print('change',next_capacility_state-capacility_state)


                    #only update user_states
                    user_states = next_user_states

                print('total reward this episode is: ', total_reward)

if __name__ == '__main__':
    devision = Devision()
    devision.train()


