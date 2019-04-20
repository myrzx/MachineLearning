% 线性回归再试
% 2019-4-18
% 拟合二次的
% 和MATLAB的走两条路了？
% 少量拟合还不错，7400轮就停了，应该是局部最低点？
% 用矩阵改写一下？

clear
close
clc
% x_origin = [168;169;170;171;172;173;174;175;176;177;178;179;180;181;182;183;184;185;186;187;188;189;190;191;192;193;194;195;168.500000000000;169.500000000000;170.500000000000;171.500000000000;172.500000000000;173.500000000000;174.500000000000;175.500000000000;176.500000000000;177.500000000000;178.500000000000;179.500000000000;180.500000000000;181.500000000000;182.500000000000;183.500000000000;184.500000000000;185.500000000000;186.500000000000;187.500000000000;188.500000000000;189.500000000000;190.500000000000;191.500000000000;192.500000000000;193.500000000000;194.500000000000;195.500000000000;168.700000000000;169.700000000000;170.700000000000;171.700000000000;172.700000000000;173.700000000000;174.700000000000;175.700000000000;176.700000000000;177.700000000000;178.700000000000;179.700000000000;180.700000000000;181.700000000000;182.700000000000;183.700000000000;184.700000000000;185.700000000000;186.700000000000;187.700000000000;188.700000000000;189.700000000000;190.700000000000;191.700000000000;192.700000000000;193.700000000000;194.700000000000;195.700000000000;167.300000000000;168.300000000000;169.300000000000;170.300000000000;171.300000000000;172.300000000000;173.300000000000;174.300000000000;175.300000000000;176.300000000000;177.300000000000;178.300000000000;179.300000000000;180.300000000000;181.300000000000;182.300000000000;183.300000000000;184.300000000000;185.300000000000;186.300000000000;187.300000000000;188.300000000000;189.300000000000;190.300000000000;191.300000000000;192.300000000000;193.300000000000;194.300000000000;195.300000000000;196.300000000000;197.300000000000;198.300000000000;199.300000000000]';
x_origin = [20 40 60 80 100 50 35];
x_2 = x_origin .^ 2;
% y = [172.360000000000;172.900000000000;172.280000000000;174.910000000000;173.400000000000;176.950000000000;175.400000000000;174.980000000000;177.450000000000;175.020000000000;178.500000000000;178.580000000000;179.500000000000;178.110000000000;179.560000000000;180.110000000000;178.600000000000;183.640000000000;181.650000000000;178.180000000000;183.630000000000;181.250000000000;183.700000000000;182.230000000000;185.750000000000;185.230000000000;186.730000000000;187.330000000000;173.360000000000;171.900000000000;176.280000000000;170.950000000000;177.400000000000;180.950000000000;170.400000000000;177.180000000000;173.650000000000;183.120000000000;172.500000000000;180.080000000000;176.500000000000;175.110000000000;180.560000000000;181.110000000000;178.600000000000;183.640000000000;177.650000000000;190.180000000000;185.630000000000;186.250000000000;180.700000000000;186.230000000000;182.750000000000;186.230000000000;182.730000000000;189.330000000000;163.360000000000;167.900000000000;174.280000000000;173.950000000000;176;183;168.900000000000;178.500000000000;171.020000000000;187.010000000000;170.980000000000;179;179.500000000000;173.110000000000;183.560000000000;179.110000000000;180.600000000000;185.640000000000;183.650000000000;192.680000000000;184.010000000000;188.250000000000;177.700000000000;188.230000000000;179.250000000000;188.890000000000;180.130000000000;193.330000000000;169.360000000000;168.360000000000;170.900000000000;175.280000000000;170.910000000000;178.400000000000;170.950000000000;180.400000000000;169.980000000000;177.450000000000;179.020000000000;175.500000000000;183.580000000000;178.500000000000;188.110000000000;189.560000000000;172.110000000000;183.600000000000;180.640000000000;190.650000000000;176.180000000000;186.630000000000;177.250000000000;189.700000000000;177.230000000000;189.750000000000;179.230000000000;189.730000000000;185.330000000000;193.330000000000;195.330000000000;189.330000000000;183.330000000000]';
y = [90 920 2490 5000 8000 1500 650];

% 均值归一化
miu1 =  mean(x_origin);
s1 = max(x_origin) - min(x_origin);
x = (x_origin - miu1) / s1;
miu2 = mean(x_2);
s2 = max(x_2) - min(x_2);
x_2_2 = (x_2 - miu2) / s2;


count = 0;      % 训练轮数
t0 = 0;         % θ0
t1 = 0;         % θ1
t2 = 0;
alpha = 1.5;   % 学习率
n = length(x);
final = 20000;
% m = 0;

while count < final
    count = count + 1;
    error = t0 + t1 * x + t2 * x_2_2 - y;
    error_0 = sum(error);         % J对θ0求导
    error_1 = sum(error .* x);  % J对θ1求导
    error_2 = sum(error .* x_2_2);
    t0 = t0 - alpha * error_0 / n;
    t1 = t1 - alpha * error_1 / n;
    t2 = t2 - alpha * error_2 / n;
    J(count) = sum(error .^ 2) / n;
    if count < 50 || final - count < 50 || mod(count, 50) == 0
        fprintf('count=%f, J=%f, t0=%f, t1=%f, t2=%f\n', ...
            count, J(count), t0 - t1 * miu1 / s1 - t2 * miu2 / s2, t1 / s1, t2 / s2);
    end
%     m = input('Please\n');
end

% 转化为原来的θ

t0 = t0 - t1 * miu1 / s1 - t2 * miu2 /s2;
t1 = t1 / s1;
t2 = t2 / s2;

x_test = 0 : 250;
y_test = t2 * (x_test .^ 2) + t1 * x_test + t0;
figure(1)
plot(x_origin, y, 'r*', x_test, y_test)
% axis([0 250 60 220])     % 控制x、y轴显示范围，[xmin，xmax，ymin，ymax]，还有set、xlim可以用
figure(2)
% p = polyfit(x_origin, y, 2);
% y_test_2 = p(1) * x_test + p(2);
p = polyfit(x_origin, y, 2);
y_test_2 = p(1) * (x_test .^ 2) + p(2) * x_test + p(3);
plot(x_origin, y, 'r*', x_test, y_test_2)
figure(3)
plot(1:final, J)
axis([0 final 0 10000])
grid