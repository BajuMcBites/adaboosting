import trainAdaboostModels as t
import adaboost as ada




if __name__ == "__main__":
    testBatch = t.unpickle("./CIFAR-10/cifar-10-batches-py/test_batch.pkl")
    [testData, testLabels] = t.combineBatches([testBatch])
    testLabels = t.convertLabels(testLabels, 1)
    testImages = t.convertDataToImages(testData, [32, 32])

                                                

    correct = 0
    postiive_correct = 0
    for index, image in enumerate(testImages):
        if (model.predict(image) < 0 and testLabels[index] == -1):
            correct += 1
        elif (model.predict(image) > 0 and testLabels[index] == 1):
            correct += 1
            postiive_correct += 1

        if (index % 1000 == 0):
            print("ACCURACY: " + str(correct/(index + 1)))
        

    print("FINAL ACCURACY: " + str(correct/len(testImages)))
    print(postiive_correct)








