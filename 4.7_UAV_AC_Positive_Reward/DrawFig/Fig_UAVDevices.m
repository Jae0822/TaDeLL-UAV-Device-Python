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


%% 数据准备
data111 = [4.547118248043992, 4.37332996299819, 3.990491207133625];
data22 = [2.806708183896269, 2.5767385689305122, 1.7828438053135873];
dataAoImean = [7.440421906419495, 7.018933338129015, 6.073652121001368];
dataCPUmean = [1.5127474088828763, 1.5920680357683799, 1.7716234552655166];
data111_data22 = [7.353826431940261, 6.950068531928702, 5.7733350124472125];
dataAoImean_dataCPUmean = [8.953169315302372, 8.611001373897395, 7.845275576266885];

%% 绘图: 
figure
hold on;
x = categorical({'Random', 'Force', 'Smart'});
x = reordercats(x,{'Random', 'Force', 'Smart'});
y = [data111(1), data22(1), dataAoImean(1), dataCPUmean(1);
    data111(2), data22(2), dataAoImean(2), dataCPUmean(2);
    data111(3), data22(3), dataAoImean(3), dataCPUmean(3)]';
bar(x,y, 1)
plot(x, data111_data22, 'ro-', 'DisplayName','Reward of Devices', 'LineWidth', 1.5) %,'Color', colorMat(7,:))
plot(x, dataAoImean_dataCPUmean, 'kx-','DisplayName','AoI of Devices', 'LineWidth', 1.5) %, 'Color', colorMat(6,:))
ylabel('Performance Values')

box on; 
grid on;

ax = gca; 
ax.FontSize = 16; 

legend('Cost of System', 'Energy of System', 'AoI of Devices', 'CPU of Devices', 'System Cost', 'Device Cost')
legend('Location','best', 'FontSize', 14)



itsokay = 1;
