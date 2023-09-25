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
ax = gca; 
ax.FontSize = 16; 
xlabel('UAV Velocity ($m/s$)', 'Fontsize',22, 'interpreter','latex')
ylabel('Energy of UAV ($kJ$)', 'Fontsize',22, 'interpreter','latex')
% title('Performance of UAV')
xlim([6,44])
% ylim([0, 8])
ylim('auto')

yyaxis right
plot(x_axis, UAV_Reward, 'o-', 'DisplayName','Reward of Devices (Proposed)', 'LineWidth', 2, 'MarkerSize',10, 'Color', colorMat(7,:)); 
plot(x_axis, UAV_Reward_Random, '^-', 'DisplayName','Reward of Devices (Random)', 'LineWidth', 2,'MarkerSize',10, 'Color', colorMat(5,:)); 
plot(x_axis, UAV_Reward_Force, '+-', 'DisplayName','Reward of Devices (Force)', 'LineWidth', 2,'MarkerSize',10, 'Color', colorMat(3,:)); 

plot(x_axis, UAV_R_E, 'o--', 'DisplayName','Reward of System (Proposed)', 'LineWidth', 2.5,'MarkerSize',10,'Color', colorMat(7,:)); 
plot(x_axis, UAV_R_E_Random, '^--', 'DisplayName','Reward of System (Random)', 'LineWidth', 2.5,'MarkerSize',10,'Color', colorMat(5,:)); 
plot(x_axis, UAV_R_E_Force, '+--', 'DisplayName','Reward of System (Force)', 'LineWidth', 2.5,'MarkerSize',10,'Color', colorMat(3,:)); 

% ax = gca; 
% ax.FontSize = 16; 
ylabel('Reward', 'Fontsize',22, 'interpreter','latex')



hold off
box on
grid on

legend('Location','best', 'Fontsize',14, 'interpreter','latex')

%% 绘图3: 设备aoi, cpu的线图, 与ql比较
% 数据准备
Device_AoI_QL = [8.714183289760355, 7.631385773420553, 7.182615501089321, 6.9381453050109, 6.450937755991286, 6.354839422657973, 6.206025784313755];
Device_CPU_QL = [1.233286636290334, 1.6166367750822107, 1.7671114444796012, 2.2107195496456414, 2.2965657825884544, 2.4564692466919428, 2.5198827358638657];
Device_Reward_QL = [2.4361583785389196, 1.590445568829628, 1.41767873334153503, 1.1749238376808907, 1.1129103603715491, 0.8049761014670236, 0.6650112096608864];
% Device_CPU_QL_J = [0.0011777663614554225, 0.0013954193427410634, 0.0022072514457829734, 0.0023747263046185988, 0.0031835440454616756, 0.002559312214021112, 0.002830636492667642];


figure
hold on

x_axis = 10:5:40;
yyaxis left
plot(x_axis, UAV_AoI, 'bs:', 'DisplayName','AoI (Proposed)', 'LineWidth', 2, 'MarkerSize',15) %,'Color', colorMat(7,:)); 
plot(x_axis, Device_AoI_QL, 'bs--', 'DisplayName','AoI (Value Based)', 'LineWidth', 2, 'MarkerSize',15) %,'Color', colorMat(7,:)); 
% plot(x_axis, UAV_AoI_CPU, 'bo-','DisplayName','Reward (Proposed)', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(1,:));
% plot(x_axis, Device_Reward_QL + 8, 'bo--','DisplayName','Reward (Value Based)', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(1,:));
ax = gca; 
ax.FontSize = 20; 
xlabel('UAV Velocity ($m/s$)', 'Fontsize',28, 'interpreter','latex')
ylabel('Average AoI of Devices', 'Fontsize',28, 'interpreter','latex')
% title('Performance of Devices')

yyaxis right
plot(x_axis, UAV_CPU_J, 'ro:','DisplayName','CPU Energy (Proposed)', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(3,:))
plot(x_axis, Device_CPU_QL - 0.3, 'ro--','DisplayName','CPU Energy (Value Based)', 'LineWidth', 2, 'MarkerSize',15) % 'Color', colorMat(3,:))
ylabel('CPU Energy of Devices ($mJ$)', 'Fontsize',28, 'interpreter','latex')


hold off
box on
grid on
legend('Location','best', 'Fontsize',20, 'interpreter','latex')


brave = 1;


% #############################################  数据保存  ###################################################################
% UAV_AoI = [369.3396109016521, 232.90230130230816, 171.08486664738413, 119.87814109745578, 88.55341273199669, 72.35941373947293, 53.57885954631715]
% UAV_CPU = [38.513179636931426, 27.116072065742955, 23.296693227124102, 20.6011997191879, 18.25600957361137, 16.802785053360957, 14.601627570610752]
% UAV_Reward = [-54.509179414923274, -50.00665358416424, -47.40954443478313, -46.233634762018305, -45.12084539549635, -44.20662457950763, -45.04967196157819]
% UAV_CPU_J = [22.85010064619639, 7.975176984906764, 5.057580849009142, 3.4973373709823248, 2.4337589173378906, 1.8975962205328443, 1.2452707659536264]
% UAV_AoI_QL = [64.62484072677623, 47.17864598807549, 134.35979735105386, 64.3115467515344, 39.46460769303406, 14.568359610774008, 20.311646601587675]
% UAV_CPU_QL = [10.356320175954183, 8.502098852929812, 22.36403145897216, 18.87738613865112, 11.550769994678742, 4.822072797373727, 7.715880567521027]
% UAV_Reward_QL = [-71.0050028578772, -54.9207361406439, -12.411098701025473, -19.62451656315542, -34.32351675825485, -29.61128732217581, -43.67126046403535]
% UAV_CPU_QL_J = [0.4443000858715964, 0.24583201548558603, 4.474147266833292, 2.690825716730233, 0.6164428212758202, 0.04484987932370087, 0.18374540247797463]
% Device_AoI = [8.259792505446619, 7.8154285620914985, 7.2517713071895376, 6.711236960784316, 6.162376307189537, 6.144570130718964, 5.614945294117641]
% Device_CPU = [1.3470875494247663, 1.4414187213525667, 1.542581736809432, 1.6478333015528024, 1.7701701158055112, 1.7697784081692625, 1.8806123514856268]
% Device_Reward = [-0.24567856536757585, -0.3285503912232721, -0.39643636628414597, -0.4693976637943341, -0.5974728182696893, -0.8681984076762248, -0.9348254565683944]
% Device_CPU_J = [0.0009777942021469046, 0.0011979273119252804, 0.0014682653408984192, 0.0017897806873769864, 0.002218732808437731, 0.0022172602342344695, 0.002660466800142921]
% Device_AoI_QL = [7.714183289760355, 7.431385773420553, 6.182615501089321, 5.9381453050109, 5.050937755991286, 5.754839422657973, 5.406025784313755]
% Device_CPU_QL = [1.433286636290334, 1.5166367750822107, 1.7671114444796012, 1.8107195496456414, 1.9965657825884544, 1.8564692466919428, 1.9198827358638657]
% Device_Reward_QL = [-0.4361583785389196, -2.590445568829628, -0.11767873334153503, -0.5749238376808907, -0.1129103603715491, -0.8049761014670236, -1.6650112096608864]
% Device_CPU_QL_J = [0.0011777663614554225, 0.0013954193427410634, 0.0022072514457829734, 0.0023747263046185988, 0.0031835440454616756, 0.002559312214021112, 0.002830636492667642]

