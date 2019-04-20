% ���Իع�������
% 2019-4-20
% �ϵ�����ã��Ҳ����ʾҲ�ܺ�
% ����Ϥ����������˷��˲���ʱ��
% mean��Ĭ�ϰ�����ƽ�����ĳ�2���ǰ���
% ����ʽ��ϣ����Ǻ�����20w�βŲ��

clear
close
clc

% ����height.txt�����ɱ���height
load E:\desktop\MATLAB_programs\height.txt

% ���ݼ�׼��
x_origin = height(:, 1); 
y = height(:, 2);
x_mean = (x_origin - mean(x_origin)) / (max(x_origin) - min(x_origin));
x_2 = x_origin .^ 2;
x_2_mean = (x_2 - mean(x_2)) / (max(x_2) - min(x_2));
x = ones(length(x_mean), 2);    % x0Ϊ1
x(:, 2) = x_mean;
x(:, 3) = x_2_mean;

count = 0;          % �ڼ���
final = 500000;        % ѵ������
t = zeros(3, 1);    % ģ�Ͳ���
alpha = 1.5;        % ѧϰ��
[m, n] = size(x);   % ��ȡ��������������������������������
J = zeros(1, m);    % ��ʧ������ֵ

% �ݶ��½�
while count < final
    count = count + 1;
    error = x * t - y;    % ע�����ά��Ҫ��Ӧ��
    t = t - alpha / m * sum(error .* x)';    % һ�仰�㶨�������£���Ԫ����������������(ά��Ϊ1���Ǹ�)��� 2*1 - (1 * 2)'���ɴ���t������������
    J(count) = sum(error .^ 2) / 2 / m;
end

Normal = x' * x \ (x') * y;   % ���淽�̽ⷨ�� x' * x \  x' = inv��x'*x��*x'
Normal(1) = Normal(1) - Normal(2) * mean(x_origin) / (max(x_origin) - min(x_origin)) - Normal(3) * mean(x_2) / (max(x_2) - min(x_2));
Normal(2) = Normal(2) / (max(x_origin) - min(x_origin));
Normal(3) = Normal(3) / (max(x_2) - min(x_2));
fprintf('Normal: t2 = %f, t1 = %f, t0 = %f\n', Normal(3), Normal(2), Normal(1));

% ת��Ϊԭ����t0��t1
t(1) = t(1) - t(2) * mean(x_origin) / (max(x_origin) - min(x_origin)) - t(3) * mean(x_2) / (max(x_2) - min(x_2));
t(2) = t(2) / (max(x_origin) - min(x_origin));
t(3) = t(3) / (max(x_2) - min(x_2));
fprintf('Gradient: After %g steps,t2 = %f, t1 = %f, t0 = %f, J = %f.\n', count, t(3), t(2), t(1), J(count));

x_test = 0 : 250;
x_t = [ones(length(x_test), 1), x_test', (x_test .^ 2)'];
y_test =  x_t * t;
figure(1)
plot(x_origin, y, 'r*', x_test, y_test, 'b')
% axis([0 250 60 220])     % ����x��y����ʾ��Χ��[xmin��xmax��ymin��ymax]������set��xlim������

figure(2)

p = polyfit(x_origin, y, 2);
y_test_2 = x_t * flipud(p');
plot(x_origin, y, 'r*', x_test, y_test_2, 'b')
fprintf('MATLAB: t2 = %f, t1 = %f, t0 = %f\n', p(1), p(2), p(3));

figure(3)
plot(1:final, J, 'r')
axis([0 final 0 J(100)])
grid