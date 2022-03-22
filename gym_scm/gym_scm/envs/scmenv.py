import gym
import numpy as np
from gym_scm.envs.utils import *
class scmEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(0, 100, shape=(4,), dtype=np.int32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,10), dtype=np.float32)
        self.RBQ = 0
        self.RQ = 1
        self.PI = 2
        self.BI = 3
        self.ABQ = 4
        self.ID = 5
        self.ED = 6
        self.AQ = 7
        self.EI = 8
        self.OOQ = 9
        self.OQ = 10
        self.BO = 11
        self.DEL_LEAD_TIME = 12
        self.storage_retailer = np.zeros((40, 13))
        self.storage_wholesaler = np.zeros((40, 13))
        self.storage_distributor = np.zeros((40, 13))
        self.storage_factory = np.zeros((40, 13))
        self.week_counter = 0
        self.order_lead_time = 0
        self.alpha = 0.25
        self.unit_holding_cost_retailer = 1
        self.unit_holding_cost_wholesaler = 1
        self.unit_holding_cost_distributor = 1
        self.unit_holding_cost_factory = 1
        self.unit_bo_cost_retailer = 2
        self.unit_bo_cost_wholesaler = 2
        self.unit_bo_cost_distributor = 2
        self.unit_bo_cost_factory = 2
        self.period_start = 1
        self.period_stop = 35
        self.state_var_list = [self.RBQ,self.RQ,self.PI,self.ABQ,self.ID,self.ED,self.AQ,self.EI,self.OOQ,self.BO]
        print("state varaibles being considered are RBQ,RQ,PI,ABQ,ID,ED,AQ,EI,OOQ,BO")

    def step(self, predictions):
        self.week_counter += 1
        assert (len(predictions) == 4)
        # predictions = np.round(predictions)
        # Factory
        # OQ
        self.storage_factory[self.week_counter - 1, self.OQ] = predictions[3]
        # OOQ
        if self.week_counter == 1:
            self.storage_factory[self.week_counter, self.OOQ] = 4 + self.storage_factory[self.week_counter - 1, self.OQ]
        # Updating RQ of Factory
        # RQ of Factory for future weeks

        self.storage_factory[
            self.week_counter + int(self.storage_retailer[self.week_counter - 1, self.DEL_LEAD_TIME]), self.RQ] = \
            self.storage_factory[
                self.week_counter + int(self.storage_retailer[self.week_counter - 1, self.DEL_LEAD_TIME]), self.RQ] + \
            self.storage_factory[self.week_counter - 1, self.OQ]
        # OOQ for future Weeks (if any)
        for i in range(0, int(self.storage_retailer[self.week_counter - 1, self.DEL_LEAD_TIME]), 1):
            if self.week_counter + i == 1:
                continue
            self.storage_factory[self.week_counter + i, self.OOQ] = self.storage_factory[
                                                                        self.week_counter + i, self.OOQ] + \
                                                                    self.storage_factory[self.week_counter - 1, self.OQ]

        # ID
        self.storage_factory[self.week_counter + self.order_lead_time, self.ID] = predictions[2]
        # RQ (special case for 2nd week) modifying current week only if it's week 2
        if self.week_counter == 1:
            self.storage_factory[self.week_counter, self.RQ] = 4 + self.storage_factory[self.week_counter, self.RQ]

        # PI
        self.storage_factory[self.week_counter, self.PI] = self.storage_factory[self.week_counter - 1, self.EI]
        # ED
        self.storage_factory[self.week_counter, self.ED] = calculateED(self.alpha,
                                                                       self.storage_factory[
                                                                           self.week_counter, self.ID],
                                                                       self.storage_factory[
                                                                           self.week_counter - 1, self.ED])
        # BI
        self.storage_factory[self.week_counter, self.BI] = calculateBI(
            self.storage_factory[self.week_counter, self.RBQ],
            self.storage_factory[self.week_counter, self.RQ],
            self.storage_factory[self.week_counter, self.PI])
        # ABQ
        self.storage_factory[self.week_counter, self.ABQ] = calculateABQ(
            self.storage_factory[self.week_counter, self.BI],
            self.storage_factory[self.week_counter, self.PI])
        # AQ
        self.storage_factory[self.week_counter, self.AQ] = calculateAQ(
            self.storage_factory[self.week_counter, self.ID],
            self.storage_factory[self.week_counter, self.BI],
            self.storage_factory[self.week_counter, self.ABQ])

        # RQ from Factory for future weeks
        self.storage_distributor[
            self.week_counter + int(self.storage_retailer[self.week_counter - 1, self.DEL_LEAD_TIME]), self.RQ] = \
            self.storage_distributor[
                self.week_counter + int(self.storage_retailer[self.week_counter - 1, self.DEL_LEAD_TIME]), self.RQ] + \
            self.storage_factory[
                self.week_counter, self.AQ]
        # RBQ from factory for future weeks
        self.storage_distributor[
            self.week_counter + int(self.storage_retailer[self.week_counter - 1, self.DEL_LEAD_TIME]), self.RBQ] = \
            self.storage_distributor[
                self.week_counter + int(self.storage_retailer[self.week_counter - 1, self.DEL_LEAD_TIME]), self.RBQ] + \
            self.storage_factory[
                self.week_counter, self.ABQ]

        # EI
        self.storage_factory[self.week_counter, self.EI] = calculateEI(
            self.storage_factory[self.week_counter, self.RBQ],
            self.storage_factory[self.week_counter, self.RQ],
            self.storage_factory[self.week_counter, self.PI],
            self.storage_factory[self.week_counter, self.ID])
        # BO
        self.storage_factory[self.week_counter, self.BO] = calculateBO(
            self.storage_factory[self.week_counter, self.EI])

        # Distributor
        # OQ
        self.storage_distributor[self.week_counter - 1, self.OQ] = predictions[2]
        # OOQ
        if self.week_counter == 1:
            self.storage_distributor[self.week_counter, self.OOQ] = 4 + self.storage_distributor[
                self.week_counter - 1, self.OQ]
        # OOQ for future weeks
        for i in range(0, int(self.storage_retailer[self.week_counter - 1, self.DEL_LEAD_TIME]), 1):
            if self.week_counter + i == 1:
                continue
            self.storage_distributor[self.week_counter + i, self.OOQ] = self.storage_distributor[
                                                                            self.week_counter + i, self.OOQ] + \
                                                                        self.storage_distributor[
                                                                            self.week_counter - 1, self.OQ]
        # ID
        self.storage_distributor[self.week_counter + self.order_lead_time, self.ID] = predictions[1]

        # RQ (special case for 2nd week) modifying current week only if it's week 2
        if self.week_counter == 1:
            self.storage_distributor[self.week_counter, self.RQ] = 4 + self.storage_distributor[
                self.week_counter, self.RQ]

        # PI
        self.storage_distributor[self.week_counter, self.PI] = self.storage_distributor[self.week_counter - 1, self.EI]
        # ED
        self.storage_distributor[self.week_counter, self.ED] = calculateED(self.alpha,
                                                                           self.storage_distributor[
                                                                               self.week_counter, self.ID],
                                                                           self.storage_distributor[
                                                                               self.week_counter - 1, self.ED])
        # BI
        self.storage_distributor[self.week_counter, self.BI] = calculateBI(
            self.storage_distributor[self.week_counter, self.RBQ],
            self.storage_distributor[self.week_counter, self.RQ],
            self.storage_distributor[self.week_counter, self.PI])

        # ABQ
        self.storage_distributor[self.week_counter, self.ABQ] = calculateABQ(
            self.storage_distributor[self.week_counter, self.BI],
            self.storage_distributor[self.week_counter, self.PI])

        # AQ
        self.storage_distributor[self.week_counter, self.AQ] = calculateAQ(
            self.storage_distributor[self.week_counter, self.ID],
            self.storage_distributor[self.week_counter, self.BI],
            self.storage_distributor[self.week_counter, self.ABQ])
        # RQ from Factory for future weeks
        self.storage_wholesaler[
            self.week_counter + int(self.storage_retailer[self.week_counter - 1, self.DEL_LEAD_TIME]), self.RQ] = \
            self.storage_wholesaler[
                self.week_counter + int(self.storage_retailer[self.week_counter - 1, self.DEL_LEAD_TIME]), self.RQ] + \
            self.storage_distributor[
                self.week_counter, self.AQ]
        # RBQ from factory for future weeks
        self.storage_wholesaler[
            self.week_counter + int(self.storage_retailer[self.week_counter - 1, self.DEL_LEAD_TIME]), self.RBQ] = \
            self.storage_wholesaler[
                self.week_counter + int(self.storage_retailer[self.week_counter - 1, self.DEL_LEAD_TIME]), self.RBQ] + \
            self.storage_distributor[
                self.week_counter, self.ABQ]

        # EI
        self.storage_distributor[self.week_counter, self.EI] = calculateEI(
            self.storage_distributor[self.week_counter, self.RBQ],
            self.storage_distributor[self.week_counter, self.RQ],
            self.storage_distributor[self.week_counter, self.PI],
            self.storage_distributor[self.week_counter, self.ID])
        # BO
        self.storage_distributor[self.week_counter, self.BO] = calculateBO(
            self.storage_distributor[self.week_counter, self.EI])

        # Wholesaler
        # OQ
        self.storage_wholesaler[self.week_counter - 1, self.OQ] = predictions[1]
        # OOQ
        if self.week_counter == 1:
            self.storage_wholesaler[self.week_counter, self.OOQ] = 4 + self.storage_wholesaler[
                self.week_counter - 1, self.OQ]
        # OOQ for future weeks
        for i in range(0, int(self.storage_retailer[self.week_counter - 1, self.DEL_LEAD_TIME]), 1):
            if self.week_counter + i == 1:
                continue
            self.storage_wholesaler[self.week_counter + i, self.OOQ] = self.storage_wholesaler[
                                                                           self.week_counter + i, self.OOQ] + \
                                                                       self.storage_wholesaler[
                                                                           self.week_counter - 1, self.OQ]
        # ID
        self.storage_wholesaler[self.week_counter + self.order_lead_time, self.ID] = predictions[0]
        # RQ (special case for 2nd week) modifying current week only if it's week 2
        if self.week_counter == 1:
            self.storage_wholesaler[self.week_counter, self.RQ] = 4 + self.storage_wholesaler[
                self.week_counter, self.RQ]

        # PI
        self.storage_wholesaler[self.week_counter, self.PI] = self.storage_wholesaler[self.week_counter - 1, self.EI]
        # ED
        self.storage_wholesaler[self.week_counter, self.ED] = calculateED(self.alpha,
                                                                          self.storage_wholesaler[
                                                                              self.week_counter, self.ID],
                                                                          self.storage_wholesaler[
                                                                              self.week_counter - 1, self.ED])
        # BI
        self.storage_wholesaler[self.week_counter, self.BI] = calculateBI(
            self.storage_wholesaler[self.week_counter, self.RBQ],
            self.storage_wholesaler[self.week_counter, self.RQ],
            self.storage_wholesaler[self.week_counter, self.PI])

        # ABQ
        self.storage_wholesaler[self.week_counter, self.ABQ] = calculateABQ(
            self.storage_wholesaler[self.week_counter, self.BI],
            self.storage_wholesaler[self.week_counter, self.PI])

        # AQ
        self.storage_wholesaler[self.week_counter, self.AQ] = calculateAQ(
            self.storage_wholesaler[self.week_counter, self.ID],
            self.storage_wholesaler[self.week_counter, self.BI],
            self.storage_wholesaler[self.week_counter, self.ABQ])
        # RQ from Factory for future weeks
        self.storage_retailer[
            self.week_counter + int(self.storage_retailer[self.week_counter - 1, self.DEL_LEAD_TIME]), self.RQ] = \
            self.storage_retailer[
                self.week_counter + int(self.storage_retailer[self.week_counter - 1, self.DEL_LEAD_TIME]), self.RQ] + \
            self.storage_wholesaler[
                self.week_counter, self.AQ]
        # RBQ from factory for future weeks
        self.storage_retailer[
            self.week_counter + int(self.storage_retailer[self.week_counter - 1, self.DEL_LEAD_TIME]), self.RBQ] = \
            self.storage_retailer[
                self.week_counter + int(self.storage_retailer[self.week_counter - 1, self.DEL_LEAD_TIME]), self.RBQ] + \
            self.storage_wholesaler[
                self.week_counter, self.ABQ]

        # EI
        self.storage_wholesaler[self.week_counter, self.EI] = calculateEI(
            self.storage_wholesaler[self.week_counter, self.RBQ],
            self.storage_wholesaler[self.week_counter, self.RQ],
            self.storage_wholesaler[self.week_counter, self.PI],
            self.storage_wholesaler[self.week_counter, self.ID])
        # BO
        self.storage_wholesaler[self.week_counter, self.BO] = calculateBO(
            self.storage_wholesaler[self.week_counter, self.EI])

        # Retailer
        # OQ
        self.storage_retailer[self.week_counter - 1, self.OQ] = predictions[0]
        # OOQ
        if self.week_counter == 1:
            self.storage_retailer[self.week_counter, self.OOQ] = 4 + self.storage_retailer[
                self.week_counter - 1, self.OQ]
        # OOQ for future weeks
        for i in range(0, int(self.storage_retailer[self.week_counter - 1, self.DEL_LEAD_TIME]), 1):
            if self.week_counter + i == 1:
                continue
            self.storage_retailer[self.week_counter + i, self.OOQ] = self.storage_retailer[
                                                                         self.week_counter + i, self.OOQ] + \
                                                                     self.storage_retailer[
                                                                         self.week_counter - 1, self.OQ]
        # ID
        # Already set as demand for retailer
        # RQ (special case for 2nd week) modifying current week only if it's week 2
        if self.week_counter == 1:
            self.storage_retailer[self.week_counter, self.RQ] = 4 + self.storage_retailer[self.week_counter, self.RQ]

        # PI
        self.storage_retailer[self.week_counter, self.PI] = self.storage_retailer[self.week_counter - 1, self.EI]
        # ED
        self.storage_retailer[self.week_counter, self.ED] = calculateED(self.alpha,
                                                                        self.storage_retailer[
                                                                            self.week_counter, self.ID],
                                                                        self.storage_retailer[
                                                                            self.week_counter - 1, self.ED])
        # RQ from Factory for future weeks
        # There is nothing downstream after retailer to set RQ
        # RBQ from factory for future weeks
        # There is nothing downstream after retailer to set RBQ

        # BI
        self.storage_retailer[self.week_counter, self.BI] = calculateBI(
            self.storage_retailer[self.week_counter, self.RBQ],
            self.storage_retailer[self.week_counter, self.RQ],
            self.storage_retailer[self.week_counter, self.PI])

        # ABQ
        self.storage_retailer[self.week_counter, self.ABQ] = calculateABQ(
            self.storage_retailer[self.week_counter, self.BI],
            self.storage_retailer[self.week_counter, self.PI])

        # AQ
        self.storage_retailer[self.week_counter, self.AQ] = calculateAQ(
            self.storage_retailer[self.week_counter, self.ID],
            self.storage_retailer[self.week_counter, self.BI],
            self.storage_retailer[self.week_counter, self.ABQ])
        # EI
        self.storage_retailer[self.week_counter, self.EI] = calculateEI(
            self.storage_retailer[self.week_counter, self.RBQ],
            self.storage_retailer[self.week_counter, self.RQ],
            self.storage_retailer[self.week_counter, self.PI],
            self.storage_retailer[self.week_counter, self.ID])
        # BO
        self.storage_retailer[self.week_counter, self.BO] = calculateBO(
            self.storage_retailer[self.week_counter, self.EI])
        #return self.storage_retailer[:35], self.storage_wholesaler[:35], self.storage_distributor[
        #                                                                 :35], self.storage_factory[:35]
        nextstate= ((self.storage_retailer[self.week_counter,self.state_var_list],self.storage_wholesaler[self.week_counter,self.state_var_list],self.storage_distributor[self.week_counter,self.state_var_list],self.storage_factory[self.week_counter,self.state_var_list]))
        r= 1*abs(max(0,nextstate[0][7]))+2*nextstate[0][9]+ 1*abs(max(0,nextstate[1][7]))+2*nextstate[1][9]+1*abs(max(0,nextstate[2][7]))+2*nextstate[2][9]+1*abs(max(0,nextstate[3][7]))+2*nextstate[3][9]
        reward =-1*r
        done=False
        info={"current_week":self.week_counter,
        "final_score":None}
        if self.week_counter >= self.period_stop:
            done=True
            info["final_score"]=self.total_cost()
            print("episode finished")
            print("final score is:",info["final_score"])
            

        return nextstate, reward, done, info
    def reset(self,
    demand=[15, 10, 8, 14, 9, 3, 13, 2, 13, 11, 3, 4, 6, 11, 15, 12, 15, 4, 12, 3, 13, 10, 15, 15, 3, 11, 1, 13, 10, 10, 0, 0, 8, 0, 14]
    , leadTime=[2, 0, 2, 4, 4, 4, 0, 2, 4, 1, 1, 0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 1, 4, 2, 2, 1, 4, 3, 4, 1, 4, 0, 3, 3, 4]):
        # First Week Conditions
        self.storage_retailer = np.zeros((40, 13))
        self.storage_wholesaler = np.zeros((40, 13))
        self.storage_distributor = np.zeros((40, 13))
        self.storage_factory = np.zeros((40, 13))
        if len(demand) != 40:
            appendArray = [0]
            for i in range(39 - len(demand)):
                appendArray.append(0)
            demand = np.append(demand, appendArray)
        if len(leadTime) != 40:
            appendArray_ = [0]
            for i in range(39 - len(leadTime)):
                appendArray_.append(0)
            leadTime = np.append(leadTime, appendArray_)
        # ORDER_LEAD_TIME
        self.week_counter = 0

        self.storage_retailer[:, self.DEL_LEAD_TIME] = leadTime
        self.storage_wholesaler[:, self.DEL_LEAD_TIME] = leadTime
        self.storage_distributor[:, self.DEL_LEAD_TIME] = leadTime
        self.storage_factory[:, self.DEL_LEAD_TIME] = leadTime

        # ID
        self.storage_retailer[:, self.ID] = demand
        self.storage_wholesaler[0, self.ID] = 4
        self.storage_distributor[0, self.ID] = 4
        self.storage_factory[0, self.ID] = 4

        # PI
        self.storage_retailer[0, self.PI] = 12
        self.storage_wholesaler[0, self.PI] = 12
        self.storage_distributor[0, self.PI] = 12
        self.storage_factory[0, self.PI] = 12

        # RBQ
        self.storage_retailer[0, self.RBQ] = 0
        self.storage_wholesaler[0, self.RBQ] = 0
        self.storage_distributor[0, self.RBQ] = 0
        self.storage_factory[0, self.RBQ] = 0

        # RQ
        self.storage_retailer[0, self.RQ] = 4
        self.storage_wholesaler[0, self.RQ] = 4
        self.storage_distributor[0, self.RQ] = 4
        self.storage_factory[0, self.RQ] = 4

        # BI
        self.storage_retailer[0, self.BI] = calculateBI(self.storage_retailer[0, self.RBQ],
                                                        self.storage_retailer[0, self.RQ],
                                                        self.storage_retailer[0, self.PI])
        self.storage_wholesaler[0, self.BI] = calculateBI(self.storage_wholesaler[0, self.RBQ],
                                                          self.storage_wholesaler[0, self.RQ],
                                                          self.storage_wholesaler[0, self.PI])
        self.storage_distributor[0, self.BI] = calculateBI(self.storage_distributor[0, self.RBQ],
                                                           self.storage_distributor[0, self.RQ],
                                                           self.storage_distributor[0, self.PI])
        self.storage_factory[0, self.BI] = calculateBI(self.storage_factory[0, self.RBQ],
                                                       self.storage_factory[0, self.RQ],
                                                       self.storage_factory[0, self.PI])

        # ABQ
        self.storage_retailer[0, self.ABQ] = calculateABQ(self.storage_retailer[0, self.BI],
                                                          self.storage_retailer[0, self.PI])
        self.storage_wholesaler[0, self.ABQ] = calculateABQ(self.storage_wholesaler[0, self.BI]
                                                            , self.storage_wholesaler[0, self.PI])
        self.storage_distributor[0, self.ABQ] = calculateABQ(self.storage_distributor[0, self.BI],
                                                             self.storage_distributor[0, self.PI])
        self.storage_factory[0, self.ABQ] = calculateABQ(self.storage_factory[0, self.BI],
                                                         self.storage_factory[0, self.PI])

        # ED
        self.storage_retailer[0, self.ED] = self.storage_retailer[0, self.ID]
        self.storage_wholesaler[0, self.ED] = self.storage_wholesaler[0, self.ID]
        self.storage_distributor[0, self.ED] = self.storage_distributor[0, self.ID]
        self.storage_factory[0, self.ED] = self.storage_factory[0, self.ID]

        # AQ
        self.storage_retailer[0, self.AQ] = calculateAQ(self.storage_retailer[0, self.ID],
                                                        self.storage_retailer[0, self.BI],
                                                        self.storage_retailer[0, self.ABQ])
        self.storage_wholesaler[0, self.AQ] = calculateAQ(self.storage_wholesaler[0, self.ID],
                                                          self.storage_wholesaler[0, self.BI],
                                                          self.storage_wholesaler[0, self.ABQ])
        self.storage_distributor[0, self.AQ] = calculateAQ(self.storage_distributor[0, self.ID],
                                                           self.storage_distributor[0, self.BI],
                                                           self.storage_distributor[0, self.ABQ])
        self.storage_factory[0, self.AQ] = calculateAQ(self.storage_factory[0, self.ID],
                                                       self.storage_factory[0, self.BI],
                                                       self.storage_factory[0, self.ABQ])

        # EI
        self.storage_retailer[0, self.EI] = calculateEI(self.storage_retailer[0, self.RBQ],
                                                        self.storage_retailer[0, self.RQ],
                                                        self.storage_retailer[0, self.PI],
                                                        self.storage_retailer[0, self.ID])
        self.storage_wholesaler[0, self.EI] = calculateEI(self.storage_wholesaler[0, self.RBQ],
                                                          self.storage_wholesaler[0, self.RQ],
                                                          self.storage_wholesaler[0, self.PI],
                                                          self.storage_wholesaler[0, self.ID])
        self.storage_distributor[0, self.EI] = calculateEI(self.storage_distributor[0, self.RBQ],
                                                           self.storage_distributor[0, self.RQ],
                                                           self.storage_distributor[0, self.PI],
                                                           self.storage_distributor[0, self.ID])
        self.storage_factory[0, self.EI] = calculateEI(self.storage_factory[0, self.RBQ],
                                                       self.storage_factory[0, self.RQ],
                                                       self.storage_factory[0, self.PI],
                                                       self.storage_factory[0, self.ID])
        # OOQ
        self.storage_retailer[0, self.OOQ] = 4
        self.storage_wholesaler[0, self.OOQ] = 4
        self.storage_distributor[0, self.OOQ] = 4
        self.storage_factory[0, self.OOQ] = 4

        # BO
        self.storage_retailer[0, self.BO] = calculateBO(self.storage_retailer[0, self.EI])
        self.storage_wholesaler[0, self.BO] = calculateBO(self.storage_wholesaler[0, self.EI])
        self.storage_distributor[0, self.BO] = calculateBO(self.storage_distributor[0, self.EI])
        self.storage_factory[0, self.BO] = calculateBO(self.storage_factory[0, self.EI])

        # RQ
        # Note the here 2 is the delivery lead time of AQ in first week
        self.storage_retailer[0 + 2, self.RQ] = self.storage_retailer[0 + 2, self.RQ] + self.storage_wholesaler[
            0, self.AQ]
        self.storage_wholesaler[0 + 2, self.RQ] = self.storage_wholesaler[0 + 2, self.RQ] + self.storage_distributor[
            0, self.AQ]
        self.storage_distributor[0 + 2, self.RQ] = self.storage_distributor[0 + 2, self.RQ] + self.storage_factory[
            0, self.AQ]

        # RBQ
        self.storage_retailer[0 + 2, self.RBQ] = self.storage_retailer[0 + 2, self.RBQ] + self.storage_wholesaler[
            0, self.ABQ]
        self.storage_wholesaler[0 + 2, self.RBQ] = self.storage_wholesaler[0 + 2, self.RBQ] + self.storage_distributor[
            0, self.ABQ]
        self.storage_distributor[0 + 2, self.RBQ] = self.storage_distributor[0 + 2, self.RBQ] + self.storage_factory[
            0, self.ABQ]

        # RQ of Factory
        self.storage_factory[2, self.RQ] = 4
        self.storage_distributor[2, self.RQ] = 4
        #return self.storage_retailer, self.storage_wholesaler, self.storage_distributor, self.storage_factory
        
        return ((self.storage_retailer[0,self.state_var_list],self.storage_wholesaler[0,self.state_var_list],self.storage_distributor[0,self.state_var_list],self.storage_factory[0,self.state_var_list]))
    def total_cost(self):
        self.total_cost_retailer = np.sum(np.maximum(
            self.storage_retailer[
            self.period_start - 1:self.period_stop, self.EI],0) * self.unit_holding_cost_retailer) + (np.sum(
            self.storage_retailer[
            self.period_start - 1:self.period_stop, self.BO] * self.unit_bo_cost_retailer))
        #print("Retailer Holding Cost:",np.sum(np.maximum(
        #   self.storage_retailer[
        #  self.period_start - 1:self.period_stop, self.EI],0) * self.unit_holding_cost_retailer))
        #print("Retailer BO cost: ",(np.sum(
        #   self.storage_retailer[
        #  self.period_start - 1:self.period_stop, self.BO] * self.unit_bo_cost_retailer)))
        self.total_cost_wholesaler = np.sum(np.maximum(
            self.storage_wholesaler[
            self.period_start - 1:self.period_stop, self.EI],0) * self.unit_holding_cost_wholesaler) + (np.sum(
            self.storage_wholesaler[
            self.period_start - 1:self.period_stop, self.BO] * self.unit_bo_cost_wholesaler))
        #print("WholeSaler Holding Cost:", np.sum(np.maximum(
        #   self.storage_wholesaler[
        #  self.period_start - 1:self.period_stop, self.EI], 0) * self.unit_holding_cost_retailer))
        #print("WholeSaler BO cost: ", (np.sum(
        #   self.storage_wholesaler[
        #  self.period_start - 1:self.period_stop, self.BO] * self.unit_bo_cost_retailer)))

        self.total_cost_distributor = np.sum(np.maximum(
            self.storage_distributor[
            self.period_start - 1:self.period_stop, self.EI],0) * self.unit_holding_cost_distributor) + (np.sum(
            self.storage_distributor[
            self.period_start - 1:self.period_stop, self.BO] * self.unit_bo_cost_distributor))
        #print("Distributor Holding Cost:", np.sum(np.maximum(
        #   self.storage_distributor[
        #  self.period_start - 1:self.period_stop, self.EI], 0) * self.unit_holding_cost_retailer))
        #print("Distributor BO cost: ", (np.sum(
        #   self.storage_distributor[
        #  self.period_start - 1:self.period_stop, self.BO] * self.unit_bo_cost_retailer)))

        self.total_cost_factory = np.sum(np.maximum(
            self.storage_factory[
            self.period_start - 1:self.period_stop, self.EI],0) * self.unit_holding_cost_factory) + (np.sum(
            self.storage_factory[
            self.period_start - 1:self.period_stop, self.BO] * self.unit_bo_cost_factory))
        #print("Factory Holding Cost:", np.sum(np.maximum(
        #   self.storage_factory[
        #  self.period_start - 1:self.period_stop, self.EI], 0) * self.unit_holding_cost_retailer))
        #print("Factory BO cost: ", (np.sum(
        #   self.storage_factory[
        #  self.period_start - 1:self.period_stop, self.BO] * self.unit_bo_cost_retailer)))
        return self.total_cost_retailer + self.total_cost_distributor+ self.total_cost_wholesaler + self.total_cost_factory
