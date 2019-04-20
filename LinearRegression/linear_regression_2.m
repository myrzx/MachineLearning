% 线性回归再试
% 2019-4-18
% 均值归一化和正规方程都会了，2333
% 均值归一化之后，效果显著！0.00553的学习率！
% 从4kw次到4k次，效率高了一万倍！特征缩放！真好用！
% 学习率变成0.0553，100次都不用！0.15，更夸张了！
% 经验，博采众长，CSDN博客上的感悟。也是先有老师的课的基础和实践动手的疑惑
% 效果好到爆炸！线性回归登堂入室！~尝试多特征和多项式！~
% 均值归一化后，α到3才爆炸，对比没有归一化之前的0.0000553，感人！
% 博客写两种，理论的和实践的！~
% 125次迭代就和MATLAB拟合的一样了

clear
close
clc

% height_id = fopen('E:\desktop\MATLAB_programs\height.txt');
% height = fread(height_id);

% 加载height.txt文件，生成变量height
load E:\desktop\MATLAB_programs\height.txt

x_origin = height(:, 1);    % 第一列？自动转成行了啊
y = height(:, 2);

x = (x_origin - mean(x_origin)) / (max(x_origin) - min(x_origin));

count = 0;      % 训练轮数
t0 = 0;         % θ0
t1 = 0;         % θ1
alpha = 1.5;   % 学习率
n = length(x);
final = 125;
% m = 0;

while count < final
    count = count + 1;
    error = t0 + t1 * x - y;
    error_0 = sum(error);         % J对θ0求导
    error_1 = sum(error .* x);  % J对θ1求导
    t0 = t0 - alpha * error_0 / n;
    t1 = t1 - alpha * error_1 / n;
    J(count) = sum(error .^ 2) / 2 / n;
    if count < 50 || final - count < 50 %mod(count, 50) == 0
        fprintf('count=%f, J=%f, t0=%f, t1=%f\n', count, J(count), t0, t1);
    end
%     m = input('Please\n');
end



% 转化为原来的t0、t1
t0 = t0 - t1 * mean(x_origin) / (max(x_origin) - min(x_origin))
t1 = t1 / (max(x_origin) - min(x_origin))

x_test = 0 : 250;
y_test = t1 * x_test + t0;
figure(1)
plot(x_origin, y, 'r*', x_test, y_test)
axis([0 250 60 220])     % 控制x、y轴显示范围，[xmin，xmax，ymin，ymax]，还有set、xlim可以用

figure(2)
p = polyfit(x_origin, y, 1);
y_test_2 = p(1) * x_test + p(2);
plot(x_origin, y, 'r*', x_test, y_test_2)

figure(3)
plot(1:final, J)
axis([0 final 0 200])
grid