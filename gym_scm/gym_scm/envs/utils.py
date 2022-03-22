
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
