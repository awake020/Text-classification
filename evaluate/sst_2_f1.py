

class sst2F1Eval(object):
    @staticmethod
    def get_socre(predict, label):
        TP = 0
        FP = 0
        FN = 0
        for p, l in zip(predict, label):
            if p == l and p == '1':
                TP += 1
            elif p == '1':
                FP += 1
            elif l == '1':
                FN += 1
        return 2 * TP / (2*TP + FP + FN)
