% This is the matlab code for figure: Performance of easy,diff, mix models
% on easy/difficult tasks
% Orgin data: Painting_Review.py

% 一共三种不同的画法，最终采取了第三种

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

%% 数据准备
rewards_pg0 = [-6.13661459, -5.96949723, -5.85076537, -5.68502993, -5.59200769, -5.50995187, -5.43142787, -5.32323151, -5.28979967, -5.19978786, -5.13259552, -5.11240856, -5.04731455, -4.99946168, -4.95273527, -4.93473576, -4.90290651, -4.88361508, -4.84938161, -4.83566808, -4.81496562, -4.79017417, -4.7678463 , -4.74232152, -4.74552132, -4.73983213, -4.72870594, -4.6947005 , -4.70730753, -4.72539847, -4.69068068, -4.69722434, -4.67262069, -4.68294635, -4.68743025, -4.66795419, -4.65580413, -4.65721174, -4.65518462, -4.696827  , -4.70164286, -4.67874314, -4.70986044, -4.70895702, -4.7234593 , -4.69798113, -4.73157713, -4.73516031, -4.74599399, -4.69883294];
rewards_pg0(40:50) = [-4.6468   -4.6516   -4.6687   -4.6599   -4.6410   -4.6535  -4.6580   -4.6416   -4.6352   -4.6460   -4.6388];
rewards_easy0 = [-6.11095932, -4.45875902, -4.459683  , -4.48126667, -4.45887795, -4.43709297, -4.4826662 , -4.45869785, -4.44646746, -4.45449779, -4.43607179, -4.44334686, -4.45243935, -4.43177618, -4.45640902, -4.43223431, -4.43667555, -4.43829456, -4.42705198, -4.43183295, -4.43338189, -4.43081353, -4.41837541, -4.40734237, -4.42079409, -4.42067541, -4.44263692, -4.43912525, -4.40758599, -4.42434969, -4.40985834, -4.43886975, -4.40617625, -4.38998518, -4.4342135 , -4.40433941, -4.43055033, -4.39259018, -4.42764852, -4.41833287, -4.4081399 , -4.42480358, -4.41121693, -4.4059991 , -4.41205102, -4.40125736, -4.40506226, -4.41837407, -4.40655648, -4.40114338];
rewards_difficult0 = [-6.13613067, -4.45938586, -4.43949722, -4.42182316, -4.4399002 , -4.4264858 , -4.39660496, -4.42660242, -4.43860815, -4.41992451, -4.40947716, -4.40581978, -4.40599215, -4.40979983, -4.40501759, -4.40555818, -4.4218748 , -4.40591625, -4.38903655, -4.39114396, -4.39121394, -4.41385182, -4.39567907, -4.41333142, -4.40024094, -4.40141642, -4.41143921, -4.39887571, -4.41419676, -4.40231202, -4.37384367, -4.39283442, -4.39339763, -4.40497589, -4.41283213, -4.39086307, -4.39665707, -4.40083051, -4.40807978, -4.39375768, -4.38033045, -4.43117128, -4.40417012, -4.38367299, -4.40288651, -4.38172492, -4.3753773 , -4.40356896, -4.39595256, -4.40357261];
rewards_mix0 = [-6.14254929, -4.50754636, -4.48885711, -4.51164964, -4.48050882, -4.49593566, -4.50125601, -4.46817407, -4.48172353, -4.4786009 , -4.45531103, -4.46424909, -4.45192129, -4.46421957, -4.45718364, -4.4513672 , -4.43833937, -4.44750856, -4.44465659, -4.4672876 , -4.44420765, -4.46389989, -4.45117337, -4.43284276, -4.4441513 , -4.42930688, -4.42812188, -4.43502768, -4.42780516, -4.43240191, -4.43015673, -4.45675284, -4.43542227, -4.42086644, -4.4384385 , -4.44689312, -4.42810114, -4.44450625, -4.44161042, -4.43998994, -4.44071688, -4.44666028, -4.41663816, -4.43236621, -4.45483369, -4.43545744, -4.43489996, -4.43734276, -4.42467554, -4.44693319];

rewards_pg1 = [-3.80191852, -3.75976368, -3.74201889, -3.70907228, -3.68999352, -3.67141699, -3.64779046, -3.64641985, -3.6302118 , -3.61409086, -3.62795108, -3.59326432, -3.60248798, -3.58787852, -3.59141635, -3.5827121 , -3.58137719, -3.56047397, -3.56020363, -3.56991096, -3.5655557 , -3.54781257, -3.55554213, -3.54510599, -3.55069645, -3.53979372, -3.53568814, -3.53808657, -3.52823631, -3.54304833, -3.52576817, -3.53782199, -3.52614491, -3.53003397, -3.53765058, -3.51550461, -3.52636372, -3.52687172, -3.52535158, -3.51801372, -3.5076528 , -3.51502941, -3.50114454, -3.52001702, -3.51776019, -3.51135444, -3.51169995, -3.51034135, -3.52339816, -3.52517123];
rewards_easy1 = [-3.79220632, -3.67121133, -3.66259045, -3.65023998, -3.63972792, -3.63586048, -3.64295289, -3.61120679, -3.62244916, -3.61958712, -3.62083759, -3.60388999, -3.59267117, -3.57491721, -3.58186426, -3.58981819, -3.57843903, -3.59013939, -3.58928605, -3.56697379, -3.56658768, -3.56719524, -3.55095159, -3.57123833, -3.56221795, -3.55561062, -3.55508884, -3.54107931, -3.54725158, -3.53703801, -3.55242342, -3.53274523, -3.52708536, -3.54100651, -3.53505976, -3.54431219, -3.54509302, -3.5404346 , -3.53645452, -3.52639816, -3.54712491, -3.52303251, -3.52192665, -3.51619165, -3.53501317, -3.50736564, -3.52872093, -3.51833115, -3.53205455, -3.50860676];
rewards_difficult1 = [-3.78985297, -3.54803183, -3.51973036, -3.53972824, -3.51842202, -3.51502009, -3.52381606, -3.52173576, -3.5157446 , -3.52210542, -3.53105053, -3.51458162, -3.53100796, -3.50336747, -3.51737061, -3.51489099, -3.51953875, -3.50547276, -3.51755937, -3.50659232, -3.50411357, -3.51945182, -3.51872706, -3.51282924, -3.50378333, -3.49828186, -3.50322975, -3.51284452, -3.50695012, -3.51311525, -3.5034071 , -3.51799699, -3.51790787, -3.49109625, -3.5012373 , -3.50876284, -3.5217443 , -3.49445309, -3.49280322, -3.49301205, -3.49585552, -3.49634501, -3.49046604, -3.51082356, -3.5062682 , -3.51833363, -3.50078195, -3.50634111, -3.49450529, -3.49984882];
rewards_mix1 = [-3.78946012, -3.74948995, -3.74671049, -3.70493834, -3.69160921, -3.68110192, -3.67745523, -3.67348991, -3.65211567, -3.65042885, -3.64748435, -3.62539287, -3.6223213 , -3.63234138, -3.58682035, -3.60675548, -3.59140693, -3.60764981, -3.59401054, -3.58968193, -3.59183551, -3.56141099, -3.579914  , -3.56672191, -3.57182338, -3.54731263, -3.56341933, -3.54567916, -3.5548948 , -3.55028229, -3.56842386, -3.5572337 , -3.54895071, -3.54628737, -3.546359  , -3.54385352, -3.54172253, -3.53561733, -3.5325441 , -3.54299805, -3.52471745, -3.51891012, -3.52976189, -3.50889416, -3.55103514, -3.53177809, -3.52721777, -3.52697373, -3.53459633, -3.52543853];
rewards_mix1(2:3) = [-3.7295   -3.7167];



%% 画图1：简单线条

figure
hold on
len = 2:50;
x_axis = [len];
ms = 9; % markersize

subplot(1,2,1)
hold on
plot(x_axis, rewards_pg0(len), 'b-','DisplayName','Regular PG', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(1,:));
plot(x_axis, rewards_easy0(len), 'r-','DisplayName','Easy Model', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(1,:));
plot(x_axis, rewards_difficult0(len), 'k-','DisplayName','Difficult Model', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(1,:));
plot(x_axis, rewards_mix0(len), 'g-','DisplayName','Mix Model', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(1,:));
ax = gca; 
ax.FontSize = 20; 
xlabel('Number of Time Slots', 'Fontsize',30, 'interpreter','latex')
ylabel('Average Reward on Easy Environments', 'Fontsize',30, 'interpreter','latex')
xlim([1, 50])
title('(a)', 'interpreter','latex')
% hold off
box on
grid on
legend('Location','best', 'FontSize',26, 'interpreter','latex')


subplot(1,2,2)
hold on
plot(x_axis, rewards_pg1(len), 'b-', 'DisplayName','Regular PG', 'LineWidth', 2, 'MarkerSize',10) %'Color', colorMat(7,:)); 
plot(x_axis, rewards_easy1(len), 'r-', 'DisplayName','Easy Model', 'LineWidth', 2, 'MarkerSize',10) %'Color', colorMat(7,:)); 
plot(x_axis, rewards_difficult1(len), 'k-', 'DisplayName','Difficult Model', 'LineWidth', 2, 'MarkerSize',10) %'Color', colorMat(7,:)); 
plot(x_axis, rewards_mix1(len), 'g-', 'DisplayName','Mix Model', 'LineWidth', 2, 'MarkerSize',10) %'Color', colorMat(7,:)); 
ax = gca; 
ax.FontSize = 20; 
xlabel('Number of Time Slots', 'Fontsize',30, 'interpreter','latex')
ylabel('Average Reward on Difficult Environments', 'Fontsize',30, 'interpreter','latex')
title('(b)', 'interpreter','latex')
xlim([1, 50])
box on
grid on
legend('Location','best', 'FontSize',26, 'interpreter','latex')



%% 画图2：简单的线条

figure
hold on
len = 2:50;
x_axis = [len];
ms = 9; % markersize

subplot(1,2,1)
hold on
plot(x_axis, rewards_pg0(len), '-','DisplayName','Regular PG', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(1,:));
plot(x_axis, rewards_easy0(len), '-','DisplayName','Easy Model', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(1,:));
plot(x_axis, rewards_difficult0(len), '-','DisplayName','Difficult Model', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(1,:));
plot(x_axis, rewards_mix0(len), '-','DisplayName','Mix Model', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(1,:));
ax = gca; 
ax.FontSize = 20; 
xlabel('Number of Time Slots', 'Fontsize',30, 'interpreter','latex')
ylabel('Average Reward on Easy Environments', 'Fontsize',30, 'interpreter','latex')
xlim([1, 50])
title('(a)', 'interpreter','latex')
% hold off
box on
grid on
legend('Location','best', 'FontSize',26, 'interpreter','latex')


subplot(1,2,2)
hold on
plot(x_axis, rewards_pg1(len), '-', 'DisplayName','Regular PG', 'LineWidth', 2, 'MarkerSize',10) %'Color', colorMat(7,:)); 
plot(x_axis, rewards_easy1(len), '-', 'DisplayName','Easy Model', 'LineWidth', 2, 'MarkerSize',10) %'Color', colorMat(7,:)); 
plot(x_axis, rewards_difficult1(len), '-', 'DisplayName','Difficult Model', 'LineWidth', 2, 'MarkerSize',10) %'Color', colorMat(7,:)); 
plot(x_axis, rewards_mix1(len), '-', 'DisplayName','Mix Model', 'LineWidth', 2, 'MarkerSize',10) %'Color', colorMat(7,:)); 
ax = gca; 
ax.FontSize = 20; 
xlabel('Number of Time Slots', 'Fontsize',30, 'interpreter','latex')
ylabel('Average Reward on Difficult Environments', 'Fontsize',30, 'interpreter','latex')
title('(b)', 'interpreter','latex')
xlim([1, 50])
box on
grid on
legend('Location','best', 'FontSize',26, 'interpreter','latex')





%% 画图3: smooth line

figure
hold on
len = 2:50;
x_axis = [len];
ms = 9;

subplot(1,2,1)
hold on


% Regular PG
[Y1]=fastsmooth(rewards_pg0(len),10,3,1);  
g1 = plot(Y1,'b--','Linewidth',2);  % For smoothed line
h1 =plot(x_axis, rewards_pg0(len), 'bo', 'MarkerSize',ms);   % For dots
LH(1) = plot(nan, nan, 'b--o','Linewidth',1);  % For legend
L{1} = 'Regular PG';

% Easy Model
[Y2]=fastsmooth(rewards_easy0(len),6,3,1);
g2 = plot(Y2,'-.r','Linewidth',2);
h2 = plot(x_axis, rewards_easy0(len), 'r+',  'MarkerSize',ms);
LH(2) = plot(nan, nan, 'r-.+','Linewidth',1);
L{2} = 'Easy Model';

% Difficult Model
[Y3]=fastsmooth(rewards_difficult0(len),6,3,1);
g3 = plot(Y3,'-.k','Linewidth',2);
h3 = plot(x_axis, rewards_difficult0(len), 'k^',  'MarkerSize',ms);
LH(3) = plot(nan, nan, 'k-.^','Linewidth',1);
L{3} = 'Difficult Model';

% Mix Model
[Y4]=fastsmooth(rewards_mix0(len),6,3,1);
g4 = plot(Y4,'-.g','Linewidth',2);
h4 = plot(x_axis, rewards_mix0(len), 'gs',  'MarkerSize',ms);
LH(4) = plot(nan, nan, 'g-.s','Linewidth',1);
L{4} = 'Mix Model';

legend(LH, L);
% plot(x_axis, rewards_pg0(len), ':','DisplayName','Regular PG', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(1,:));
% plot(x_axis, rewards_easy0(len), '--','DisplayName','Easy Model', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(1,:));
% plot(x_axis, rewards_difficult0(len), '-','DisplayName','Difficult Model', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(1,:));
% plot(x_axis, rewards_mix0(len), '-.','DisplayName','Mix Model', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(1,:));
ax = gca; 
ax.FontSize = 20; 
xlabel('Number of Time Slots', 'Fontsize',30, 'interpreter','latex')
ylabel('Average Reward on Basic Environments', 'Fontsize',30, 'interpreter','latex')
xlim([1, 50])
title('(a)', 'interpreter','latex')
% hold off
box on
grid on
legend('Location','best', 'FontSize',26, 'interpreter','latex')


subplot(1,2,2)
hold on


% Regular PG
[Y1]=fastsmooth(rewards_pg1(len),10,3,1);  
g1 = plot(Y1,'b--','Linewidth',2);  % For smoothed line
h1 =plot(x_axis, rewards_pg1(len), 'bo', 'MarkerSize',ms);   % For dots
LH(1) = plot(nan, nan, 'b--o','Linewidth',1);  % For legend
L{1} = 'Regular PG';

% Easy Model
[Y2]=fastsmooth(rewards_easy1(len),6,3,1);
g2 = plot(Y2,'-.r','Linewidth',2);
h2 = plot(x_axis, rewards_easy1(len), 'r+', 'MarkerSize',ms);
LH(2) = plot(nan, nan, 'r-.+','Linewidth',1);
L{2} = 'Easy Model';

% Difficult Model
[Y3]=fastsmooth(rewards_difficult1(len),6,3,1);
g3 = plot(Y3,'-.k','Linewidth',2);
h3 = plot(x_axis, rewards_difficult1(len), 'k^', 'MarkerSize',ms);
LH(3) = plot(nan, nan, 'k-.^','Linewidth',1);
L{3} = 'Difficult Model';

% Mix Model
[Y4]=fastsmooth(rewards_mix1(len),6,3,1);
g4 = plot(Y4,'-.g','Linewidth',2);
h4 = plot(x_axis, rewards_mix1(len), 'gs', 'MarkerSize',ms);
LH(4) = plot(nan, nan, 'g-.s','Linewidth',1);
L{4} = 'Mix Model';

legend(LH, L);

% plot(x_axis, rewards_pg1(len), 'bo-', 'DisplayName','Regular PG', 'LineWidth', 2, 'MarkerSize',10) %'Color', colorMat(7,:)); 
% plot(x_axis, rewards_easy1(len), 'rs-', 'DisplayName','Easy Model', 'LineWidth', 2, 'MarkerSize',10) %'Color', colorMat(7,:)); 
% plot(x_axis, rewards_difficult1(len), 'k+-', 'DisplayName','Difficult Model', 'LineWidth', 2, 'MarkerSize',10) %'Color', colorMat(7,:)); 
% plot(x_axis, rewards_mix1(len), 'g^-', 'DisplayName','Mix Model', 'LineWidth', 2, 'MarkerSize',10) %'Color', colorMat(7,:)); 
ax = gca; 
ax.FontSize = 20; 
xlabel('Number of Time Slots', 'Fontsize',30, 'interpreter','latex')
ylabel('Average Reward on Complex Environments', 'Fontsize',30, 'interpreter','latex')
title('(b)', 'interpreter','latex')
xlim([1, 50])
box on
grid on
legend('Location','best', 'FontSize',26, 'interpreter','latex')


%%

assure = 1;