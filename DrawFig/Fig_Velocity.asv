% This is the matlab code for figure: performance related to UAV's velocity
% Orgin data: test5.py

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
UAV_AoI = [8.210503014525013,
 7.269328281316031,
 6.923566142213826,
 6.386839525514192,
 6.275775044650731,
 6.162776722696884,
 6.05088530730253];
UAV_CPU_J = [0.9951046181458372,
 1.4728955943969633,
 1.6793564976113333,
 2.0455089381373073,
 2.115526956239253,
 2.204856932602479,
 2.2858043982141383];
UAV_AoI_CPU = [9.565493513927741,
 8.81352985464933,
 8.536788760901459,
 8.10968818801824,
 8.018061354225228,
 7.929248925297086,
 7.838715871485892];

%% 绘图1: 设备aoi, cpu等的线图
figure
hold on

x_axis = 10:5:40;
yyaxis left
plot(x_axis, UAV_AoI, 'rs--', 'DisplayName','AoI of Devices', 'LineWidth', 2, 'MarkerSize',15) %,'Color', colorMat(7,:)); 
plot(x_axis, UAV_AoI_CPU, 'bo--','DisplayName','Reward of Devices', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(1,:));
xlabel('UAV Velocity ($m/s$)', 'Fontsize',20, 'interpreter','latex')
ylabel('Average AoI and Reward of Devices', 'Fontsize',20, 'interpreter','latex')
% title('Performance of Devices')

yyaxis right
plot(x_axis, UAV_CPU_J, 'K*--','DisplayName','CPU Energy of Devices', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(3,:))
ylabel('Devices CPU Energy Consumption ($mJ$)', 'Fontsize',20, 'interpreter','latex')


hold off
box on
grid on
legend('Location','best', 'Fontsize',22, 'interpreter','latex')


%% 数据准备
UAV_Energy = [3.2886853724150815,
 2.33013145285238,
 2.702666173674077,
 3.0766007696778286,
 4.130849569897612,
 5.672599482307396,
 7.795204091580909];
UAV_Reward = [4.866173750745929,
 4.47765382643898,
 4.311336346041963,
 4.116271725786353,
 4.073836892975343,
 3.8519072630807627,
 3.7924980368158705];
UAV_Reward_Random = [5.16255110385256,
 4.848547716600551,
 4.693354142430065,
 4.4354239815633125,
 4.1946914810856875,
 4.149487046005461,
 4.095344103691454];
UAV_Reward_Force = [5.068597740323952,
 4.749025406577276,
 4.638582356676227,
 4.371589544121557,
 4.176900823274315,
 4.093122711880476,
 4.027656046851857];
UAV_R_E = [4.077429561580506,
 3.40389263964568,
 3.50700125985802,
 3.596436247732091,
 4.102343231436477,
 4.66225337269408,
 5.788380577113248];
UAV_R_E_Random = [4.334827435588925,
 3.7986309942443572,
 3.8796082758014854,
 3.9909821470348863,
 4.357210207189581,
 5.339129071722063,
 6.007323265099282];
UAV_R_E_Force = [4.335315423420666,
 3.737209697333393,
 3.82208565076736,
 3.8564407374530076,
 4.220412565052778,
 5.184555307471753,
 5.9150932382516865];

%% 绘图2: 系统能量表现等的柱状图与线图
figure
hold on
x_axis = 10:5:40;
yyaxis left
bar(x_axis, (UAV_Energy * 1000),'DisplayName','Energy of the UAV')
xlabel('UAV Velocity ($m/s$)', 'Fontsize',20, 'interpreter','latex')
ylabel('Energy of UAV ($kJ$)', 'Fontsize',20, 'interpreter','latex')
% title('Performance of UAV')
xlim([6,44])
% ylim([0, 8])
ylim('auto')

yyaxis right
plot(x_axis, UAV_Reward, 'o-', 'DisplayName','Reward of Devices (Proposed)', 'LineWidth', 2.5, 'MarkerSize',10, 'Color', colorMat(7,:)); 
plot(x_axis, UAV_Reward_Random, '^-', 'DisplayName','Reward of Devices (Random)', 'LineWidth', 2.5,'MarkerSize',10, 'Color', colorMat(5,:)); 
plot(x_axis, UAV_Reward_Force, '*-', 'DisplayName','Reward of Devices (Force)', 'LineWidth', 2.5,'MarkerSize',10, 'Color', colorMat(3,:)); 

plot(x_axis, UAV_R_E, 'o--', 'DisplayName','Reward of System (Proposed)', 'LineWidth', 2.5,'MarkerSize',10,'Color', colorMat(7,:)); 
plot(x_axis, UAV_R_E_Random, '^--', 'DisplayName','Reward of System (Random)', 'LineWidth', 2.5,'MarkerSize',10,'Color', colorMat(5,:)); 
plot(x_axis, UAV_R_E_Force, '*--', 'DisplayName','Reward of System (Force)', 'LineWidth', 2.5,'MarkerSize',10,'Color', colorMat(3,:)); 


ylabel('Reward of System and Devices', 'Fontsize',20, 'interpreter','latex')

hold off
box on
grid on

legend('Location','best', 'Fontsize',14, 'interpreter','latex')

brave = 1;
