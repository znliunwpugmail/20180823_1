import datetime
import random
import numpy as np
import operator as op
import time
np.set_printoptions(suppress=True)


class User_Create():
    def __init__(self):
        self.alluser_sequence = []
        self.alluser_action = []
        self.alluser_action = []
        self.currentuser_sequence = []

    def Time_Leave(self,nowTime,time_s):
        time_date,time_hsm = nowTime.split(' ')
        h,m,s = time_hsm.split(':')
        s_i = int(s)+time_s
        m_i = int(m)+s_i//60
        h_i = int(h)+m_i//60
        s_i = s_i%60
        m_i = m_i%60
        h_i = h_i%24
        time_hsm = str(h_i)+":"+str(m_i).zfill(2)+":"+str(s_i).zfill(2)
        leaveTime = time_date+" "+time_hsm
        return leaveTime

    def Imitate_User(self,attributes_num = 3,max_time_interval = 60):
        "App SMS Channel_broadcast Drainage_online Reservation_callback Artificial_hotline"
        users_num = random.randint(100,1000)
        users_num = 500
        Station0 = np.ones(shape=[users_num,attributes_num+1],dtype=np.float)
        seq = list(range(0, 1000))
        users_id = random.sample(seq,users_num)
        users_id = np.array(users_id)
        Station0[:,0]=users_id

        time_speed = [random.randint(10,50) for _ in range(users_num)]

        for i in range(1,attributes_num):
            f_random = [random.randint(0,100) for _ in range(users_num)]
            f_random = np.array(f_random)/100
            Station0[:,i]=f_random
        Station0_list = Station0.tolist()
        Station0_list_result=[]

        for i in range(len(Station0_list)):
            Station0_list_i = Station0_list[i]
            time_s = time_speed[i]
            nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            Station0_list_i.append(nowTime)
            leaveTime = self.Time_Leave(nowTime,time_s)
            Station0_list_i.append(leaveTime)
            Station0_list_result.append(Station0_list_i)

        self.currentuser_sequence = Station0_list_result

        return np.array(Station0_list_result)

def sleeptime(hour,min,sec):
    return hour*3600 + min*60 + sec;

if __name__ == '__main__':
    second = sleeptime(0,0,4);
    user_create = User_Create()
    while 1==1:
        time.sleep(1)
        if len(user_create.alluser_sequence) > 0:
            nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for user_attributes in user_create.alluser_sequence:
                user_leavetime = user_attributes[-1]
                if op.gt(nowTime, user_leavetime):
                    user_create.alluser_sequence.remove(user_attributes)
        user_create.Imitate_User()
        user_create.alluser_sequence.extend(user_create.currentuser_sequence)