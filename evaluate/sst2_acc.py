class sst2AccEval(object):
    @staticmethod
    def get_score(predict, label):
        T = 0
        F = 0
        for p, l in zip(predict, label):
            if p == l:
                T += 1
            else :
                F += 1
        return 1.0 * T / (T + F)
