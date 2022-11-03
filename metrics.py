def computeAUROC(dataGT, dataPRED, classCount):
    outAUROC = []
    dataGT = np.array(dataGT)
    dataPRED = np.array(dataPRED)
    for i in range(classCount):
        try:
            outAUROC.append(roc_auc_score(dataGT[:, i], dataPRED[:, i]))
        except:
            outAUROC.append(0.)
    return outAUROC