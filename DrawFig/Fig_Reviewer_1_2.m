% This is the matlab code for figure: shuffling of environments order
% Orgin data: Painting_Review.py

clear all;
close all;
%% Color Map
% https://ww2.mathworks.cn/help/matlab/ref/matlab.graphics.axis.axes-properties.html#budumk7_sep_shared-ColorOrder
% https://ww2.mathworks.cn/help/matlab/ref/matlab.graphics.axis.axes-properties.html#budumk7_sep_shared-ColorOrder
% colorMat = [    0    0.4470    0.7410
% 0.8500    0.3250    0.0980
% 0.9290    0.6940    0.1250
% 0.4940    0.1840    0.5560
% 0.4660    0.6740    0.1880
% 0.3010    0.7450    0.9330
% 0.6350    0.0780    0.1840];

% ModelKeyWord = ["Diff", "Easy", "Easy_Diff", "Diff_Easy"]

%% 数据准备
ModelLine_Diff = [-2696.3515583375056,
 -3937.515744528982,
 -4150.564009524617,
 -4121.926098861923,
 -4091.5959239266153,
 -4243.8039467257295,
 -4246.18135013702,
 -4246.020295229963,
 -4245.539358304858,
 -4244.680245149647,
 -4243.2062134536,
 -4180.633454551445,
 -3906.241479621572,
 -3175.1010902075645,
 -3265.184461088929,
 -3099.405143705061,
 -1157.9341301625288,
 -193.14236122155796,
 -193.14236122155796,
 -168.4822462918088,
 -166.36838956855908,
 -171.35411100784017,
 -168.4822462918088,
 -166.36838956855908,
 -164.05335024507988];
ModelLine_Easy = [-2829.189672049785,
 -4884.727378288411,
 -4691.485355817576,
 -4696.42861077496,
 -4686.485607375599,
 -4634.052356342046,
 -4728.7697764513405,
 -4693.337776309273,
 -4694.258291617261,
 -4683.811526916244,
 -4183.215115259163,
 -4406.711453140342,
 -4724.6231600731,
 -4309.256941599647,
 -3703.7840407141475,
 -4157.098255285163,
 -2718.511323537429,
 -782.8097972238434,
 -389.8313094257386,
 -221.28721450836844,
 -219.0239965636886,
 -223.1474830688924,
 -244.3208504395443,
 -213.24566705784633,
 -223.1474830688924];
ModelLine_Easy_Diff = [-2505.26349047068,
 -4866.650363028413,
 -3878.2847734112584,
 -3846.41385121169,
 -3879.105628506153,
 -3853.3635365103946,
 -4402.3816548572095,
 -4153.866089332181,
 -4331.27239525112,
 -4257.670194916184,
 -4287.197103859143,
 -3953.8655822076266,
 -1093.5589557474768,
 -151.88706017374827,
 -184.71913291814576,
 -173.6465958448812,
 -170.39581158472825,
 -166.80446421003643,
 -176.96979895377328,
 -173.24916616592745,
 -165.58153737119,
 -181.834041732245,
 -166.1126211226175,
 -177.94864951098748,
 -179.35268727125165];
ModelLine_Diff_Easy = [-2816.4350326467566,
 -3952.8786893202105,
 -4816.627478169266,
 -4918.713408203093,
 -4968.597262510344,
 -4740.247150268101,
 -3881.5385265406508,
 -4652.597614492722,
 -4121.869742435914,
 -4141.189398410966,
 -3998.139060245651,
 -4349.731583460574,
 -4012,
 -5029.2494196459165,
 -5188.005009450967,
 -4457.494436300015,
 -4342.687369162201,
 -4329.957424091211,
 -4619.861598738259,
 -4140.622279478832,
 -4336.444237161571,
 -4394.204936049846,
 -1934.115641221983,
 -198.53203683152313,
 -195.9257884315187];


Data.reward = [6.369624586333131,
  7.009156669446991,
  8.544597896761807,
  6.972700821613074]; % type = ("Easy", "Diff", "EasyDiff", "DiffEasy")
Data.AoI = [7.080716078431372,
  4.2454890196078425,
  8.683595686274511,
  6.019809411764705];
Data.CPU = [5.689462313864629,
  9.092980311373049,
  8.33997397140887,
  7.757106103193186];
Data.b = [25.72248760073313,
  47.27234857003625,
  68.45488552223492,
  53.09679707950818];

%% 画图：柱状图
figure
hold on 

%% reward
subplot(2,2,1)
x = categorical({'Easy', 'Diff', 'EasyDiff', 'DiffEasy'});
x = reordercats(x,{'Easy', 'Diff', 'EasyDiff', 'DiffEasy'});
y = [Data.reward(1); Data.reward(2); Data.reward(3); Data.reward(4)];
bar(x,y)
ylabel('Average Reward', 'FontSize', 14, 'interpreter','latex')
set(gca,'TickLabelInterpreter','latex', 'Fontsize',13);
title('(a)', 'interpreter','latex')
grid on
%% AoI
subplot(2,2,2)
y = [Data.AoI(1); Data.AoI(2); Data.AoI(3); Data.AoI(4)];
bar(x,y)
ylabel('Average AoI', 'FontSize', 14, 'interpreter','latex')
set(gca,'TickLabelInterpreter','latex', 'Fontsize',13);
title('(b)', 'interpreter','latex')
grid on
%% CPU
subplot(2,2,3)
y = [Data.CPU(1); Data.CPU(2); Data.CPU(3); Data.CPU(4)];
bar(x,y)
ylabel('Average CPU Energy ($mJ$)', 'FontSize', 14, 'interpreter','latex')
set(gca,'TickLabelInterpreter','latex', 'Fontsize',13);
title('(c)', 'interpreter','latex')
grid on
%% b
subplot(2,2,4)
y = [Data.b(1); Data.b(2); Data.b(3); Data.b(4)];
bar(x,y)
ylabel('Average Queue Length', 'FontSize', 14, 'interpreter','latex')
set(gca,'TickLabelInterpreter','latex', 'Fontsize',13);
title('(d)', 'interpreter','latex')
grid on



%% 画图：线图
% ModelKeyWord = ["Diff", "Easy", "Easy_Diff", "Diff_Easy"]
% b r k g

figure
hold on
x_axis = 1:25;
ms = 9; % markersize
Nd = 25;

easy = ModelLine_Easy/Nd;
diff = ModelLine_Diff/Nd;
easy_diff = ModelLine_Easy_Diff/Nd;
diff_easy = ModelLine_Diff_Easy/Nd;

easy_diff(19:25) =  [-7.0788 -6.7300 -6.6233 -6.6734 -6.6445 -6.6547 -6.5621];
diff(19:25) =       [-7.7257 -6.9393 -6.8547 -6.8542 -7.0393 -7.1179 -7.1741];

hold on
plot(x_axis, easy, '+-','DisplayName','Scenario 1', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(1,:));
plot(x_axis, diff, 's-','DisplayName','Scenario 2', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(1,:));
plot(x_axis, easy_diff, 'o-','DisplayName','Scenario 3', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(1,:));
plot(x_axis, diff_easy, '^-','DisplayName','Scenario 4', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(1,:));
ax = gca; 
ax.FontSize = 20; 
xlabel('Number of Episodes', 'Fontsize',30, 'interpreter','latex')
ylabel('Average Reward', 'Fontsize',30, 'interpreter','latex')
xlim([1, 25])
% title('(a)', 'interpreter','latex')
box on
grid on
legend('Location','best', 'FontSize',26, 'interpreter','latex')

% -7.1741
% -8.9259

d = 1;






