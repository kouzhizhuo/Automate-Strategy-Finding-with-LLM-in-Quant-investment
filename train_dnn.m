%% 清除环境
clear;
close all;

%% 读取目标数据
profit_data = readmatrix('result/profit.csv');
profit_alpha = profit_data(1:end, 2);

%% 计算 min 和 max
min_val = min(profit_alpha);
max_val = max(profit_alpha);

%% 计算 0 对应的标准化值
normalized_zero = (0 - min_val) / (max_val - min_val);

disp(['标准化后 0 对应的值为: ', num2str(normalized_zero)]);

%profit_alpha = normalize(profit_alpha, 'range');

%% 读取输入数据
input_files = dir('result/alpha/*.csv');
num_files = length(input_files);
input_matrix = [];

for i = 1:num_files
    file_path = fullfile(input_files(i).folder, input_files(i).name);
    alpha_data = readmatrix(file_path);
    alpha_values = alpha_data(1:end-1, 2); % 删去最后一行数据
    alpha_values = normalize(alpha_values, 'range');
    input_matrix = [input_matrix; alpha_values'];
end

%% 构建输入和输出矩阵
P = input_matrix;
T = profit_alpha';

%% 划分数据集为训练和测试集
[trainInd, testInd] = dividerand(size(P, 2), 0.8, 0.2);
P_train = P(:, trainInd);
T_train = T(:, trainInd);
P_test = P(:, testInd);
T_test = T(:, testInd);

%% 定义神经网络
hiddenLayerSize = 1;
net = patternnet(hiddenLayerSize);
net.trainParam.epochs = 10000;
net.trainParam.goal = 1e-06;
net.trainParam.lr = 100;
%net.initFcn = 'initlay';
%net.layers{1}.initFcn = 'initwb';
%net.inputWeights{1,1}.initFcn = 'randsmall';
%net.biases{1}.initFcn = 'randsmall';
%net.performParam.regularization = 0.1;
net.trainParam.max_fail = 10000;

%% 训练神经网络
net = train(net, P_train, T_train);

%% 测试神经网络
outputs_train = net(P_train);
outputs_test = net(P_test);
accuracy_train = sum(round(outputs_train) == T_train) / numel(T_train);
accuracy_test = sum(round(outputs_test) == T_test) / numel(T_test);
disp(['训练精度: ', num2str(accuracy_train*100), '%']);
disp(['测试精度: ', num2str(accuracy_test*100), '%']);

%% 保存网络
genFunction(net, 'DNN1.m');
