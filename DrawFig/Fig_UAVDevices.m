% This is the matlab code for figure: basic comparison between three
% methods, and the UAV and Devices AoI and CPU.
% Orgin data: test.py


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


%% Data Preparation from test_new.py
UAV_Energy = [2.4253602137127084, 2.1704162859226597, 0.09199626287251937]; 
UAV_R_E =  [3.0299302108645407, 2.8382339759244055, 1.4851536317395007];
UAV_Reward = [3.634500208016373, 3.5060516659261496, 2.8783110006064807];


%% 绘图共享y坐标轴: 
figure
hold on;
x = categorical({'Random', 'Force', 'AC'});
x = reordercats(x,{'Random', 'Force', 'AC'});
% y = [UAV_Reward(1), UAV_R_E(1), UAV_Energy(1);
%     UAV_Reward(2), UAV_R_E(2), UAV_Energy(2);
%     UAV_Reward(3), UAV_R_E(3), UAV_Energy(3);];
y = [UAV_R_E(1), UAV_Energy(1),UAV_Reward(1);
    UAV_R_E(2), UAV_Energy(2), UAV_Reward(2);
    UAV_R_E(3), UAV_Energy(3), UAV_Reward(3)];
bar(x,y, 1)
% plot(x, data111_data22, 'ro-', 'DisplayName','Reward of Devices', 'LineWidth', 1.5) %,'Color', colorMat(7,:))
% plot(x, dataAoImean_dataCPUmean, 'kx-','DisplayName','AoI of Devices', 'LineWidth', 1.5) %, 'Color', colorMat(6,:))
ylabel('Reward of System and Devices')

box on; 
grid on;

ax = gca; 
ax.FontSize = 16; 

legend('Reward of the System',  'Energy of the UAV (MJ)', 'Reward of the Devices')
% legend('Reward of the Devices', 'Reward of the System', 'Energy of the UAV (MJ)')
legend('Location','best', 'FontSize', 14)



itsokay = 1;

%% Double y-axis
figure
hold on;
x = categorical({'Random', 'Force', 'AC'});
x = reordercats(x,{'Random', 'Force', 'AC'});
y = [UAV_R_E(1), UAV_Reward(1);
    UAV_R_E(2), UAV_Reward(2);
    UAV_R_E(3), UAV_Reward(3)];
z = [UAV_Energy(1);
    UAV_Energy(2);
    UAV_Energy(3)] * 1000 + 2200;
nil = [0;0;0];
bar(x, [y,nil], 'grouped')
ylabel('Reward of System and Devices', 'Fontsize',20, 'interpreter','latex')
% legend('Reward of the System', 'Reward of the Devices')
set(gca,'TickLabelInterpreter','latex', 'Fontsize',20);

yyaxis right
bar(x, [nil, nil, z], 'grouped', 'FaceColor', '#EDB120');
ylabel('Energy of the UAV (kJ)', 'Fontsize',20, 'interpreter','latex')

box on; 
grid on;

ax = gca; 
ax.FontSize = 16; 

legend('Reward of the System', 'Reward of the Devices', 'Energy of the UAV')
% legend('Energy of the UAV (kJ)')
legend('Location','best', 'FontSize', 16, 'interpreter','latex')

d = 1;


