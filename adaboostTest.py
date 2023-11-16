import trainAdaboostModels as t
import adaboost as ada




if __name__ == "__main__":
    test_batch = t.unpickle("./CIFAR-10/cifar-10-batches-py/test_batch.pkl")
    [test_data, test_labels] = t.combineBatches([test_batch])
    test_images = t.convertDataToImages(test_data, [32, 32])

    model_names = ["models/model_0.pkl", "models/model_1.pkl", "models/model_2.pkl", "models/model_3.pkl", "models/model_4.pkl", "models/model_5.pkl", "models/model_6.pkl", "models/model_7.pkl", "models/model_8.pkl", "models/model_9.pkl"]
    models = [ada.ImAdaBoost(0, 0) for i in range(len(model_names))]
    for i in range(10):
        models[i].reconstruct_model(model_names[i])

    correct = 0
    for i in range(len(test_images)):
        preds = []
        for model in models:
            preds.append(model.predict(test_images[i]))
        
        max_pred = max(preds)
        prediction = preds.index(max_pred)

        if (prediction == test_labels[i]):
            correct += 1

        if (i + 1) % 1000 == 0:
            print ("ACCURACY SO FAR: " + str(correct / (i + 1)))

    print ("FINAL ACCURACY: " + str(correct / len(test_images)))

    
    #CODE FOR TESTING INDIVIDUAL CLASSIFIER
    # correct = 0
    # postiive_correct = 0
    # for index, image in enumerate(testImages):
    #     if (model.predict(image) < 0 and testLabels[index] == -1):
    #         correct += 1
    #     elif (model.predict(image) > 0 and testLabels[index] == 1):
    #         correct += 1
    #         postiive_correct += 1

    #     if (index % 1000 == 0):
    #         print("ACCURACY: " + str(correct/(index + 1)))
        
    # print("FINAL ACCURACY: " + str(correct/len(test_images)))
    # print(postiive_correct)








