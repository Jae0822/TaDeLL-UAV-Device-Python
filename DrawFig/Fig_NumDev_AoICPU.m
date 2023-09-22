% This is the matlab code to draw figure: AoI and CPU v.s. number of
% Devices. Orgin data: test6.py
clear

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
Device_Reward_1 = [4.020734877102882,
 5.818885857020991,
 6.008456957106583,
 6.245746263286723,
 6.383528169034918,
 6.404543389066634];
Device_AoI_1 = [4.810474607843139,
 8.398535921568627,
 8.864993862745099,
 9.083474669235954,
 9.498326957860337,
 9.809756817496229];
Device_CPU_J = [9.867702411266302,
 4.669866892344854,
 4.494714826568241,
 4.426975708331607,
 4.389789500313073,
 4.391664995554752];


%% 开始画图
figure
hold on

x_axis = 5:5:30;

yyaxis left
plot(x_axis, Device_Reward_1, 'bo--', 'DisplayName','Reward of Devices', 'LineWidth', 2, 'MarkerSize',15) %'Color', colorMat(7,:)); 
plot(x_axis, Device_AoI_1, 'rs--','DisplayName','AoI of Devices', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(1,:));
xlabel('Number of Devices', 'Fontsize',20, 'interpreter','latex')
ylabel('Average AoI and Reward of Devices', 'Fontsize',20, 'interpreter','latex')
% title('Performance of Devices')



yyaxis right
plot(x_axis, Device_CPU_J, 'k*--','DisplayName','CPU of Devices', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(3,:))
ylabel('Devices CPU Energy Consumption ($mJ$)', 'Fontsize',20, 'interpreter','latex')
hold off
box on
grid on

legend('Location','best', 'FontSize',22, 'interpreter','latex')

brave = 1;