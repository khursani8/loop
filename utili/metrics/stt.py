import jiwer

def wer():
    def cal_wer(preds,labels):
        return jiwer.wer(labels,preds)
    return cal_wer

def cer():
    def cal_cer(preds,labels):
        return jiwer.cer(labels,preds)
    return cal_cer