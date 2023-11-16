data_batch1 = load("CIFAR-10/cifar-10-batches-mat/data_batch_1.mat");
data_batch2 = load("CIFAR-10/cifar-10-batches-mat/data_batch_2.mat");
data_batch3 = load("CIFAR-10/cifar-10-batches-mat/data_batch_3.mat");
data_batch4 = load("CIFAR-10/cifar-10-batches-mat/data_batch_4.mat");
data_batch5 = load("CIFAR-10/cifar-10-batches-mat/data_batch_5.mat");

test_batch = load("CIFAR-10/cifar-10-batches-mat/test_batch.mat");

testData = test_batch.data;
testLabels = test_batch.labels;


data_batches = [data_batch1, data_batch2, data_batch3, data_batch4, data_batch5];
[trainingData, labels] = combineBatches (data_batches);


trainingData = double(trainingData);
testData = double(testData);
% 
% test1 = test_data(1, :);
% size(test1);
% label1 = test_labels(1)

P = projectionMatrix(trainingData);
trainingData = trainingData * P;
testData = testData * P;

correct = 0;
for i = 1 : s(1)
    testImage = testData(i, :);
    class = KNN(trainingData, labels, testImage);
    if class == testLabels(i)
        correct = correct + 1;
        c = class
        percentCorrect = correct / i
        i
    end
end


function P = projectionMatrix (trainingData)
    meanTrainingData = mean(trainingData);
    stdTrainingData = std(trainingData);
    centeredTrainingData = (trainingData - meanTrainingData) ./ stdTrainingData;
    covarianceMatrix = cov(centeredTrainingData);
    [U, ~, ~] = svd(covarianceMatrix);
    size(U)
    P = U(:, 1:255);
end




% KNN (trainingData, labels, test1)



% data_batches = [data_batch1]

% pic = data(2, :);
% 
% pic = reshape(pic, [32, 32, 3]);
% 
% imshow(pic);

% P = getP(data_batches);





function classification = KNN (trainingData, labels, im)
    K = 10;
    distanceVector = getDistanceVector (trainingData, im);
    kLowestIndices = getKLowestIndices(distanceVector, K);

    classifications = zeros([10, 1]);
    for i = 1 : K
        classifications(i) = labels(kLowestIndices(i));
    end

    classification = mode(classifications);
end

function kLowestIndices = getKLowestIndices(vector, k)
    [~, sortedIndices] = sort(vector);
    kLowestIndices = sortedIndices(1:k);
end

function distanceVector = getDistanceVector (trainingData, im)
    s = size(trainingData);
    distanceVector = zeros([s(1), 1]);
    parfor i = 1 : s(1)
        distanceVector(i) = norm(trainingData(i, :)' - im');
    end   
end

function [A, labels] = combineBatches (data_batches)
    s = size(data_batches);
    A = [];
    labels = [];
    
    for i = 1 : s(2)
        A = [A; data_batches(i).data];
        labels = [labels; data_batches(i).labels];
    end
    size(labels)
    size(A)
end

% function P = getP (data_batches)
%     A = combineBatches (data_batches);
%     % A = normalizeMatrix (A); %TRY REMOVING THIS AFTER AND SEE HOW RESULTS CHANGE
%     A = double(A);
%     [U, S, V] = svd(A);
% 
% 
%     size(U)
% end
% 
% function normA = normalizeMatrix(A)
%     s = size(A);
%     normA = zeros(s);
%     for i = 1 : s(2)
%         normA(:, i) = A(:, i) - mean(A(:, i));
%         mean(A(:, i))
%         mean(normA(:, i))  
%     end
% end
% 
% function distance = vectorDistance(v1, v2)
%     distance = 0;
%     s = size(v1);
% 
%     for i = 1 : s(2)
%         distance = distance + (v1(i) - v2(i))^2;
%     end
% 
%     distance = sqrt(distance);
% end




