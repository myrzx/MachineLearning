% ���Իع�����
% 2019-4-18
% ��ֵ��һ�������淽�̶����ˣ�2333
% ��ֵ��һ��֮��Ч��������0.00553��ѧϰ�ʣ�
% ��4kw�ε�4k�Σ�Ч�ʸ���һ�򱶣��������ţ�����ã�
% ѧϰ�ʱ��0.0553��100�ζ����ã�0.15���������ˣ�
% ���飬�����ڳ���CSDN�����ϵĸ���Ҳ��������ʦ�ĿεĻ�����ʵ�����ֵ��ɻ�
% Ч���õ���ը�����Իع�������ң�~���Զ������Ͷ���ʽ��~
% ��ֵ��һ���󣬦���3�ű�ը���Ա�û�й�һ��֮ǰ��0.0000553�����ˣ�
% ����д���֣����۵ĺ�ʵ���ģ�~
% 125�ε����ͺ�MATLAB��ϵ�һ����

clear
close
clc

% height_id = fopen('E:\desktop\MATLAB_programs\height.txt');
% height = fread(height_id);

% ����height.txt�ļ������ɱ���height
load E:\desktop\MATLAB_programs\height.txt

x_origin = height(:, 1);    % ��һ�У��Զ�ת�����˰�
y = height(:, 2);

x = (x_origin - mean(x_origin)) / (max(x_origin) - min(x_origin));

count = 0;      % ѵ������
t0 = 0;         % ��0
t1 = 0;         % ��1
alpha = 1.5;   % ѧϰ��
n = length(x);
final = 125;
% m = 0;

while count < final
    count = count + 1;
    error = t0 + t1 * x - y;
    error_0 = sum(error);         % J�Ԧ�0��
    error_1 = sum(error .* x);  % J�Ԧ�1��
    t0 = t0 - alpha * error_0 / n;
    t1 = t1 - alpha * error_1 / n;
    J(count) = sum(error .^ 2) / 2 / n;
    if count < 50 || final - count < 50 %mod(count, 50) == 0
        fprintf('count=%f, J=%f, t0=%f, t1=%f\n', count, J(count), t0, t1);
    end
%     m = input('Please\n');
end



% ת��Ϊԭ����t0��t1
t0 = t0 - t1 * mean(x_origin) / (max(x_origin) - min(x_origin))
t1 = t1 / (max(x_origin) - min(x_origin))

x_test = 0 : 250;
y_test = t1 * x_test + t0;
figure(1)
plot(x_origin, y, 'r*', x_test, y_test)
axis([0 250 60 220])     % ����x��y����ʾ��Χ��[xmin��xmax��ymin��ymax]������set��xlim������

figure(2)
p = polyfit(x_origin, y, 1);
y_test_2 = p(1) * x_test + p(2);
plot(x_origin, y, 'r*', x_test, y_test_2)

figure(3)
plot(1:final, J)
axis([0 final 0 200])
grid