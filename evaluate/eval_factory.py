from evaluate.sst2_acc import sst2AccEval
from evaluate.sst_2_f1 import sst2F1Eval


class EvalFactory(object):
    @staticmethod
    def get_eval(type_name):
        if type_name == 'acc':
            return sst2AccEval()
        elif type_name == 'f1':
            return sst2F1Eval()