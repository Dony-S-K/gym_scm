#PPO implementation in SCM

import numpy as np
import pandas as pd
import tensorflow as tf
import gym
import math
from matplotlib import pyplot as plt 
import time

tf.compat.v1.disable_eager_execution()
""""""
print(tf.__version__)


# Calculations
def calculateBI(RBQ, RQ, PI):
    return RBQ + RQ + max(0, PI)


def calculateABQ(BI, PI):
    if PI >= 0:
        return 0
    elif PI < 0 and BI > 0:
        if BI >= abs(PI):
            return abs(PI)
        else:
            return BI
    elif PI < 0 and BI == 0:
        return 0


def calculateAQ(ID, BI, ABQ):
    if (BI - ABQ) >= ID:
        return max(0, ID)
    else:
        return max(0, BI - ABQ)


def calculateEI(RBQ, RQ, PI, ID):
    return (RBQ + RQ + PI) - ID


def calculateBO(EI):
    return abs(min(0, EI))


def calculateED(alpha, ID, ED):
    return int((alpha * ID) + ((1 - alpha) * ED))


class supplychain_sim:
    def __init__(self):
        # (number of weeks, number of columns)
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
    def reset(self, demand, leadTime):
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
    def step_one_week(self, predictions):
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
        return ((self.storage_retailer[self.week_counter,self.state_var_list],self.storage_wholesaler[self.week_counter,self.state_var_list],self.storage_distributor[self.week_counter,self.state_var_list],self.storage_factory[self.week_counter,self.state_var_list]))
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

    def save(self,path):
        self.retailer_df = pd.DataFrame(np.round(self.storage_retailer),
                                        columns=["RBQ", "RQ", "PI", "BI", "ABQ", "ID", "ED", "AQ", "EI", "OOQ", "OQ", "BO", "LEAD_TIME"])
        self.wholesaler_df = pd.DataFrame(np.round(self.storage_wholesaler),
                                          columns=["RBQ", "RQ", "PI", "BI", "ABQ", "ID", "ED", "AQ", "EI", "OOQ", "OQ", "BO", "LEAD_TIME"])
        self.distributor_df = pd.DataFrame(np.round(self.storage_distributor),
                                           columns=["RBQ", "RQ", "PI", "BI", "ABQ", "ID", "ED", "AQ", "EI", "OOQ", "OQ", "BO", "LEAD_TIME"])
        self.factory_df = pd.DataFrame(np.round(self.storage_factory),
                                       columns=["RBQ", "RQ", "PI", "BI", "ABQ", "ID", "ED", "AQ", "EI", "OOQ", "OQ", "BO", "LEAD_TIME"])
        self.retailer_df.to_csv(path+'retailer.csv')
        self.wholesaler_df.to_csv(path+'wholesaler.csv')
        self.distributor_df.to_csv(path+'distributor.csv')
        self.factory_df.to_csv(path+'factory.csv')


class ppo():
    def __init__(self,name,s_dim,a_dim,memory,a_bound):
        self.s_dim = s_dim
        self.a_dim =a_dim
        self.memory = memory
        self.a_bound =a_bound
        self.name = name
        self.policy  = self.make_policy()
        self.critic = self.make_critic()
        
    def make_policy(self):
            

        state_inputs = tf.keras.Input(shape=(self.s_dim,), name='state')
        advantage = tf.keras.Input(shape=(1, ), name="Advantage")
        action= tf.keras.Input(shape=(self.a_dim,), name="action")
        x = tf.keras.layers.Dense(200, activation='relu')(state_inputs)
        x1 = tf.keras.layers.Dense(200, activation='relu')(x)
        mu_0 = tf.keras.layers.Dense(self.a_dim, activation='tanh')(x1)
        x2 = tf.keras.layers.Dense(30, activation='relu')(x)
        sigma_0 = tf.keras.layers.Dense(self.a_dim, activation='relu')(x2)

        mu = tf.keras.layers.Lambda(lambda x:tf.clip_by_value(x * self.a_bound,-self.a_bound ,self.a_bound) )(mu_0)
        #mu = tf.keras.layers.Reshape((1,self.a_dim))(mu)

        covari = tf.keras.layers.Lambda(lambda x:tf.clip_by_value(x,1e-2 , 1e+02))(sigma_0)
        #covari = tf.keras.layers.Reshape((1,self.a_dim))(covari)
        
        #concat=custom_concat()
        mucov=tf.keras.layers.concatenate([mu,covari],axis=-1)
        #mucov =  tf.keras.layers.Lambda(lambda x:tf.keras.backend.expand_dims(x,1))(mucov)
        
        #mucov=concat([mu,covari])

        def proximal_policy_optimization_loss(advantage, action):
            loss_clipping = 0.2
            entropy_loss = 0.0
            pi=3.1415926
            k=self.a_dim

            def loss(y_true,y_pred):
                
                mu =y_pred[:,:k]#batch,k
                cov = y_pred[:,k:]#batch,k
                old_mu = y_true[:,:k]
                old_cov = y_true[:,k:]
                x= action

                det = tf.keras.backend.prod(cov,axis=1,keepdims=True)
                inv = 1/cov#tf.linalg.inv(cov_mat)
                norm_const = 1.0/ ( tf.keras.backend.pow(2*pi,k/2) * tf.keras.backend.pow(det,1.0/2) )
                x_mu = x - mu
                #tf.print("deter",det)
                #tf.print(" cov inv_cov",cov,inv)
                #tf.print("x,mu,xmu",x,mu,x_mu)
                x_mu_sq = tf.keras.backend.square(x_mu) 
                prod  = inv*x_mu_sq
                #tf.print('x_mu_sq',x_mu_sq)
                #tf.print('prod inv*x_mu_sq',prod,tf.keras.backend.int_shape(prod))

                prod2 =tf.keras.backend.sum(prod,axis=1,keepdims=True) 
                #tf.print("prod2 elemt sum ",prod2,tf.keras.backend.int_shape(prod2))
                result = tf.keras.backend.exp( -0.5 *prod2)
                pdf = norm_const * result
                
                old_det = tf.keras.backend.prod(old_cov,axis=1,keepdims=True)
                ##tf.print("old deter",old_det)
                old_inv = 1/old_cov#tf.linalg.old_inv(old_cov_mat)
                old_norm_const = 1.0/ ( tf.keras.backend.pow(2*pi,k/2) * tf.keras.backend.pow(old_det,1.0/2) )
                old_x_mu = x - old_mu
                #tf.print("old_cov old_inv_old_cov",old_cov,old_inv)
                #tf.print("x,old_mu,xold_mu",x,old_mu,old_x_mu)
                old_x_mu_sq = tf.keras.backend.square(old_x_mu) 
                old_prod  = old_inv*old_x_mu_sq
                #tf.print('old_x_mu_sq',old_x_mu_sq)
                #tf.print('old_prod oldinv*oldx_mu_sq',old_prod,tf.keras.backend.int_shape(old_prod))

                old_prod2 =tf.keras.backend.sum(old_prod,axis=1,keepdims=True) 
                #print("old_prod2 elemt sum ",old_prod2,tf.keras.backend.int_shape(old_prod2))
                old_result = tf.keras.backend.exp( -0.5 *old_prod2)
                old_pdf = old_norm_const * old_result                   

                    

                log_pdf = tf.keras.backend.log(pdf + tf.keras.backend.epsilon())
                
                old_log_pdf = tf.keras.backend.log(old_pdf + tf.keras.backend.epsilon() )
                entropy =  0.5 * (tf.keras.backend.log(2. * pi * det) + 1.)

                
                r = tf.keras.backend.exp(log_pdf- old_log_pdf)
                loss = -tf.keras.backend.mean(tf.keras.backend.minimum(r * advantage, tf.keras.backend.clip(r, min_value=1 - loss_clipping,max_value=1 + loss_clipping) * advantage) + entropy_loss *entropy)
                
                return loss
            return loss 
        policy= tf.keras.Model(inputs=(state_inputs, advantage,action), outputs=mucov, name='p_actor_model')
        policy.compile(loss=proximal_policy_optimization_loss(advantage=advantage,action=action), optimizer=tf.keras.optimizers.Adam(lr=0.0001))
        return policy

    def make_critic(self):
        state_inputs = tf.keras.Input(shape=(self.s_dim,), name='state')

        x = tf.keras.layers.Dense(200, activation='relu')(state_inputs)
        x = tf.keras.layers.Dense(200, activation='relu')(x)
        value_outputs = tf.keras.layers.Dense(1, activation=None)(x)
        critic= tf.keras.Model(inputs=state_inputs, outputs=value_outputs, name='p_critic_model')
        critic.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.0005), metrics=['accuracy'])
        return critic
    def save_weights(self,path):
        actorpath=path+self.name+"actor.h5"
        criticpath=path+self.name+"critic.h5"
        self.policy.save_weights(actorpath)
        self.critic.save_weights(criticpath)
        print("saved")
    def load_weights(self,path):
        actorpath=path+self.name+"actor.h5"
        criticpath=path+self.name+"critic.h5"
        self.policy.load_weights(actorpath)
        self.critic.load_weights(criticpath)
        print("loaded")

    def gae_calc(self,val,val_,rew,done):
        mask=1 
        gae=0
        gamma=0.96
        lambd = 0.95
        returns=np.zeros_like(val)
        for i in reversed(range(0,len(val))):
            mask=1
            if done[i]:
                mask = 0    
            delta=rew[i]+gamma*val_[i]*mask - val[i]
            gae=delta+gamma*lambd*mask*gae
            returns[i]=gae+val[i]
        return returns
        
    def adv_calc(self,val,val_,rew,done):
        gamma=0.96
        returns=np.zeros_like(val)
        for i in range(0,len(val)):
            returns[i] = rew[i] + (1- done[i])*val_[i]*gamma
        return returns
    def train(self,batch=512,epochs=10):
        obs =np.array( self.memory.batch_s)
        values = self.critic.predict(np.array(self.memory.batch_s))
        values_ = self.critic.predict(np.array(self.memory.batch_s_))
        returns = self.gae_calc(values,values_,self.memory.batch_r,self.memory.batch_done)#
        advantage=returns-values
        Action=np.array(self.memory.batch_a)
        Old_Prediction_musig =np.array(self.memory.musig) 
        Old_cov = np.array(self.memory.cov)
        #print("policy_train")
        self.policy.fit(x=(obs,advantage,Action),y=Old_Prediction_musig,batch_size=batch,shuffle=True, epochs=epochs, verbose=False)
        #print("value_train")
        self.critic.fit([obs],[returns], batch_size=batch, shuffle=True, epochs=epochs, verbose=False)
        self.memory.clear()

class Memory:
    def __init__(self):
        self.batch_s = []
        self.batch_a = []
        self.batch_r = []
        self.batch_s_ = []
        self.batch_done = []
        self.musig = []
        self.cov=[]
    def store(self, s, a, s_, r, done,musig,cov):
        self.batch_s.append(s)
        self.batch_a.append(a)
        self.batch_r.append(r)
        self.batch_s_.append(s_)
        self.batch_done.append(done)
        self.musig.append(musig)
        self.cov.append(cov)
    def clear(self):
        self.batch_s.clear()
        self.batch_a.clear()
        self.batch_r.clear()
        self.batch_s_.clear()
        self.batch_done.clear()
        self.musig.clear()
        self.cov.clear()
    def cnt_samples(self):
        return len(self.batch_s)

class state_buffer():
    def __init__(self,state_shape,size):
        self.buff=np.zeros((size,state_shape))
        self.state_shape = state_shape
        self.size =size
    def append(self,state):
        self.buff[0:self.size-1] = self.buff[1:self.size]  
        self.buff[self.size-1] = state
    def rec_buff(self):
        ret = np.copy(self.buff)
        return ret
    def flat_buff(self):
        ret =np.copy(self.buff.ravel())
        return ret
    def reset(self):
        self.buff=np.zeros((self.size,self.state_shape))
             
def std_scale(state):
    mu= np.array([3.4762250100302716,4.698279353948294,-6.333449063703899,3.7492747733823433,8.5,7.611111111111111,4.448673534602488,-6.658944699725241,16.367622877274282,12.031716301901945,
        3.75217063326474,4.943187078838415,-3.539219139357051,3.8486351686469993,8.99232448903687,6.957497222222222,4.837856837967346,-3.8361859162909697,17.102301267940177,11.585352109296117,
        3.2340433469019234,5.932188464648834,-0.9096331675948385,4.044458294183929,9.18653049249012,7.273116666666667,4.942966161014052,-0.9299318485340791,17.9631954189408,10.449251822634054,0.0,
        9.69554150822154,5.258817665912071,3.4779561051916543,9.69646956183448,7.647477777777778,6.0392712846910195,5.257889612299155,18.643078359069882,7.216604791940109,])

    sig = np.array([8.070845474012103,7.055978961359058,21.95859264179756,6.881778022940847,5.166666666666666,2.6170146029581773,5.448919191086436,22.254951565438294,10.754572545426946,
        15.966778789270737,8.714267232347298,7.546250941334377,24.51288978042613,7.794235362852792,6.35654410062446,2.2468624203803382,5.917721491839479,24.877633815882334,13.54679866814289,
        16.736033405314473,7.964900535674789,8.94041688262012,25.104668249979376,8.402824595641802,7.265217150757013,2.716495200670594,6.249540127866824,25.58490125382911,14.540818244533346,
        16.04104142978307,0.001,11.667680326209178,23.806329733516716,7.779764705612392,8.014605105353622,3.0851400606699007,6.9136094092318086,24.27598823845265,15.46254790676004,12.902716693510907,])
    x = np.asarray(state).reshape(1,10*4).ravel()
    out  = (x-mu)/sig
    #print(out,out.shape)
    return out 


default_demand = [15, 10, 8, 14, 9, 3, 13, 2, 13, 11, 3, 4, 6, 11, 15, 12, 15, 4, 12, 3, 13, 10, 15, 15, 3, 11, 1, 13, 10, 10, 0, 0, 8, 0, 14]
delivery_lead_time = [2, 0, 2, 4, 4, 4, 0, 2, 4, 1, 1, 0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 1, 4, 2, 2, 1, 4, 3, 4, 1, 4, 0, 3, 3, 4]


x = [] 

y = [] 
env = supplychain_sim() ##Supply_Chain_Environment
NAME = "03"################<<omly this
s_dim = 10*4
print(s_dim)
a_dim = 4
print(a_dim)
a_bound =1

DUMMY_ACTION, DUMMY_VALUE = np.zeros((1,a_dim)), np.zeros((1, 1))


time_horizon=4 #### time look back 4 works


path = '/content/gdrive/My Drive/Reinforcement Learning/RL simulator/Experiment 3/exp3/'

memory_=Memory()
agent =  ppo(name =NAME,s_dim=s_dim*time_horizon ,a_dim= a_dim,memory = memory_,a_bound=a_bound)
st_buffer = state_buffer(state_shape=s_dim,size=time_horizon)
start_time = time.time()

best = 8000
episodes =200000
steps = 35
test_ep = 1000
stop_thres=10
print(a_bound)
stop_flag=False
test_ctr=0
for ep in range(0,episodes):
    done=False
    state = env.reset(default_demand, delivery_lead_time)
    s= std_scale(state)
    st_buffer.reset()
    st_buffer.append(s)
    st_b = st_buffer.flat_buff() 
    if stop_flag:
        if test_ctr>test_ep:
            break
        else:
            test_ctr+=1    
    for step in range(steps):

        reqs = np.array([state[0][4],state[1][4],state[2][4],state[3][4]])#np.array([s[0][4],s[1][4],s[2][4],s[3][4]])
        out= agent.policy.predict((np.array([st_b]),DUMMY_VALUE,DUMMY_ACTION))[0]
        mu_pred,cov_pred =out[0:a_dim],out[a_dim:]

        cov_pred = np.diag(cov_pred)

        action= np.random.multivariate_normal(mu_pred,cov_pred,a_dim)[0]
        a =np.clip(action,-a_bound,a_bound )
        act_env =abs(np.round(10*a)+reqs)#reqs+(-2,2)
        act_env = np.where(act_env<0,0,act_env)#negative orders are not possible
        nextstate = env.step_one_week(act_env)
        s_= std_scale(nextstate)
        st_buffer.append(s_)
        st_b_ = st_buffer.flat_buff()
        reward =1*abs(max(0,nextstate[0][7]))+2*nextstate[0][9]+ 1*abs(max(0,nextstate[1][7]))+2*nextstate[1][9]+1*abs(max(0,nextstate[2][7]))+2*nextstate[2][9]+1*abs(max(0,nextstate[3][7]))+2*nextstate[3][9]
        r =-1*reward
        if step == steps-1:
            done =True
        #print(action,a,act_env,r)
    
        agent.memory.store(st_b.ravel(),action,st_b_.ravel(),r,done,out,cov_pred)# s, a, s_, r, done,musig
        
        s=s_
        state = nextstate
        st_b=st_b_
    # updation
    print("| episode: "+str(ep)+"| total Cost is : " +str(np.sum(env.total_cost()))+" |"+"time_horizon :"+str(time_horizon)+" |"+"best :"+str(best) )
    if np.sum(env.total_cost())<best:
        best = np.sum(env.total_cost())
        agent.save_weights(path)
        env.save(path)

    print("_"*100)
    #print('training')
    x.append(ep)
    y.append(best) 
    if np.sum(env.total_cost())<stop_thres:
        stop_flag=True  
    if ep % 100 == 0 :
        agent.train(batch=64)
#print("demand: " ,demand_fixed_game(NAME))   
print("best score : ", best)
total_time = time.time() - start_time
# plotting the points  
print("total time taken :",total_time/60  ," mins")
plt.plot(x, y) 
  
# naming the x axis 
plt.xlabel('No. of Episodes') 
# naming the y axis 
plt.ylabel('Total SC Cost') 
  
# giving a title to my graph 
plt.title("scm buffer ppo "+NAME+"|min: "+str(best)+" , last 100 avg "+str(np.mean(np.array(y)[-100:])))
#plt.title(str(demand_fixed_game(NAME))) 
plt.savefig(path+NAME+" total time taken"+str(total_time/60)+'mins.png')    
# function to show the plot 
plt.show() 


