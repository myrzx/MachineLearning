% 线性回归向量化
% 2019-4-20
% 断点真好用，右侧的提示也很好
% 不熟悉矩阵操作，浪费了不少时间
% mean，默认按列求平均。改成2就是按行
% 多项式拟合，还是很慢，20w次才差不多

clear
close
clc

% 加载height.txt，生成变量height
load E:\desktop\MATLAB_programs\height.txt

% 数据集准备
x_origin = height(:, 1); 
y = height(:, 2);
x_mean = (x_origin - mean(x_origin)) / (max(x_origin) - min(x_origin));
x_2 = x_origin .^ 2;
x_2_mean = (x_2 - mean(x_2)) / (max(x_2) - min(x_2));
x = ones(length(x_mean), 2);    % x0为1
x(:, 2) = x_mean;
x(:, 3) = x_2_mean;

count = 0;          % 第几轮
final = 500000;        % 训练轮数
t = zeros(3, 1);    % 模型参数
alpha = 1.5;        % 学习率
[m, n] = size(x);   % 获取矩阵行数和列数，即样本数和特征数
J = zeros(1, m);    % 损失函数的值

% 梯度下降
while count < final
    count = count + 1;
    error = x * t - y;    % 注意矩阵维数要对应！
    t = t - alpha / m * sum(error .* x)';    % 一句话搞定参数更新（四元数。。。），逐列(维度为1的那个)相乘 2*1 - (1 * 2)'。干脆让t等于行向量？
    J(count) = sum(error .^ 2) / 2 / m;
end

Normal = x' * x \ (x') * y;   % 正规方程解法。 x' * x \  x' = inv（x'*x）*x'
Normal(1) = Normal(1) - Normal(2) * mean(x_origin) / (max(x_origin) - min(x_origin)) - Normal(3) * mean(x_2) / (max(x_2) - min(x_2));
Normal(2) = Normal(2) / (max(x_origin) - min(x_origin));
Normal(3) = Normal(3) / (max(x_2) - min(x_2));
fprintf('Normal: t2 = %f, t1 = %f, t0 = %f\n', Normal(3), Normal(2), Normal(1));

% 转化为原来的t0、t1
t(1) = t(1) - t(2) * mean(x_origin) / (max(x_origin) - min(x_origin)) - t(3) * mean(x_2) / (max(x_2) - min(x_2));
t(2) = t(2) / (max(x_origin) - min(x_origin));
t(3) = t(3) / (max(x_2) - min(x_2));
fprintf('Gradient: After %g steps,t2 = %f, t1 = %f, t0 = %f, J = %f.\n', count, t(3), t(2), t(1), J(count));

x_test = 0 : 250;
x_t = [ones(length(x_test), 1), x_test', (x_test .^ 2)'];
y_test =  x_t * t;
figure(1)
plot(x_origin, y, 'r*', x_test, y_test, 'b')
% axis([0 250 60 220])     % 控制x、y轴显示范围，[xmin，xmax，ymin，ymax]，还有set、xlim可以用

figure(2)

p = polyfit(x_origin, y, 2);
y_test_2 = x_t * flipud(p');
plot(x_origin, y, 'r*', x_test, y_test_2, 'b')
fprintf('MATLAB: t2 = %f, t1 = %f, t0 = %f\n', p(1), p(2), p(3));

figure(3)
plot(1:final, J, 'r')
axis([0 final 0 J(100)])
grid