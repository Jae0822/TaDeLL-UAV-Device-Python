% This is the matlab code for figure: LLRL v.s. Regular PG
% Orgin data: test4.py

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



%% 绘图: （2，2）四幅柱状图对比
figure
hold on 
ND = 25; % number of devices for average purpose

%% reward
subplot(2,2,1)
Reward_Random =  115.47413077281708;
Reward_Force = 110.49724936199449;
Reward_Samrt = 102.33108273730522;
Reward_Random_Regular = 131.20693973099355;
Reward_Force_Regular = 129.6263582571714;
Reward_Samrt_Regular = 119.94441507408624;
x = categorical({'Random', 'Force', 'Proposed'});
x = reordercats(x,{'Random', 'Force', 'Proposed'});
y = [Reward_Random, Reward_Random_Regular; Reward_Force, Reward_Force_Regular; Reward_Samrt, Reward_Samrt_Regular]/ND;
bar(x,y)
legend('Lifelong RL', 'Regular PG')
legend('Location','best', 'FontSize', 14, 'interpreter','latex')
ylabel('Average Reward', 'FontSize', 14, 'interpreter','latex')
set(gca,'TickLabelInterpreter','latex', 'Fontsize',13);
title('(a)')
grid on

% ax = gca; 
% ax.FontSize = 12; 


%% AoI
subplot(2,2,2)
AoI_Random = 190.5572561764706;
AoI_Force = 178.33940343137257;
AoI_Smart = 135.14116931372553;
AoI_Random_Regular = 230.87788225490192;
AoI_Force_Regular = 227.67450254901962;
AoI_Smart_Regular = 175.90957568627456;
x = categorical({'Random', 'Force', 'Proposed'});
x = reordercats(x,{'Random', 'Force', 'Proposed'});
y = [AoI_Random, AoI_Random_Regular; AoI_Force, AoI_Force_Regular; AoI_Smart, AoI_Smart_Regular]/ND;
bar(x,y)
legend('Lifelong RL', 'Regular PG')
legend('Location','best' ,'FontSize', 14, 'interpreter','latex')
ylabel('Average AoI' ,'FontSize', 14, 'interpreter','latex')
title('(b)')
grid on
set(gca,'TickLabelInterpreter','latex', 'Fontsize',13);
% ax = gca; 
% ax.FontSize = 12; 


%% CPU
subplot(2,2,3)
CPU_Random = 36.7380836308312;
CPU_Force = 35.33583338167071;
CPU_Smart = 30.52640570684251;
CPU_Random_Regular = 26.883224704986183;
CPU_Force_Regular = 27.27044515980158;
CPU_Smart_Regular = 27.044576793881596;
x = categorical({'Random', 'Force', 'Proposed'});
x = reordercats(x,{'Random', 'Force', 'Proposed'});
y = [CPU_Random, CPU_Random_Regular; CPU_Force, CPU_Force_Regular; CPU_Smart, CPU_Smart_Regular]/ND;
bar(x,y)
% ylim('auto')
ylim([0, 1.7])
legend('Lifelong RL', 'Regular PG', 'interpreter','latex')
legend('Location','best' ,'FontSize', 14)
ylabel('Average CPU Energy ($mJ$)', 'FontSize', 14, 'interpreter','latex')
set(gca,'TickLabelInterpreter','latex', 'Fontsize',13);
title('(c)')
grid on

% ax = gca; 
% ax.FontSize = 12; 

%% b
subplot(2,2,4)
b_Random = 1472.20781287722;
b_Force = 1367.709765934574;
b_Smart = 870.7518711630732;
b_Random_Regular = 1859.2979398662033;
b_Force_Regular = 1845.4415908067724;
b_Smart_Regular = 1280.3929608107703;
x = categorical({'Random', 'Force', 'Proposed'});
x = reordercats(x,{'Random', 'Force', 'Proposed'});
y = [b_Random, b_Random_Regular; b_Force, b_Force_Regular; b_Smart, b_Smart_Regular]/ND;
bar(x,y)
legend('Lifelong RL', 'Regular PG')
legend('Location','best' ,'FontSize', 14, 'interpreter','latex')
ylabel('Average Queue Length', 'FontSize', 14, 'interpreter','latex')
set(gca,'TickLabelInterpreter','latex', 'Fontsize',13);
title('(d)')
grid on
% ax = gca; 
% ax.FontSize = 12; 


doit = 1;