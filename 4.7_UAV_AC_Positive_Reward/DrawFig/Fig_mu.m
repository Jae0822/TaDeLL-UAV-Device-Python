% This is the matlab code for figure: different mu 
% Orgin data: test3.py


clear;
%% Color Map
% https://ww2.mathworks.cn/help/matlab/ref/matlab.graphics.axis.axes-properties.html#budumk7_sep_shared-ColorOrder
% https://ww2.mathworks.cn/help/matlab/ref/matlab.graphics.axis.axes-properties.html#budumk7_sep_shared-ColorOrder
colorMat = [    0    0.4470    0.7410
0.8500    0.3250    0.0980
0.9290    0.6940    0.1250
0.4940    0.1840    0.5560
0.4660    0.6740    0.1880
0.3010    0.7450    0.9330
0.6350    0.0780    0.1840];


%% 数据准备
avg = [249.28786980883305, 189.15444325937256, 152.18987930465954];
sumreward = [327.22027898495725, 238.51654150244985, 213.59731877126382];
sumenergy = [146.19319203571416, 138.2640050240418, 135.6791795484068];


%% 绘图: 
figure
% x = [0.5, 0.7, 0.9];
x = categorical({'0.5', '0.7', '0.9'});
y = [avg(1), sumreward(1), sumenergy(1); avg(2), sumreward(2), sumenergy(2); avg(3), sumreward(3), sumenergy(3)];
bar(x,y, 1)
xlabel('Value of \mu')
ylabel('Performance of the System')
hold off
box on
grid on
legend('Totale Reward of the System', 'Reward of the Devices', 'Energy of the UAV')
legend('Location','best')

brave = 1;


