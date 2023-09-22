period = 20;
load('Case_8_Learn.mat');

PGELLA = Avg_Learn_AbPGmodel_Mat(4,1:period);
PGELLA = [PGELLA Avg_Learn_AbPGmodel_Mat(5,1:40)];
PGELLA = [PGELLA Avg_Learn_AbPGmodel_Mat(6,1:period)];
PGELLA = [PGELLA Avg_Learn_AbPGmodel_Mat(7,1:period)];

load('Case_5_All.mat')
Regular_PG = Avg_r(4,1:period);
load('Case_6_All.mat')
Regular_PG = [Regular_PG  Avg_r(5,1:40)];
load('Case_7_All.mat')
Regular_PG = [Regular_PG  Avg_r(6,1:period)];
load('Case_8_All.mat')
Regular_PG = [Regular_PG  Avg_r(7,1:period)];

figure
length = size(PGELLA,2); % make sure they have the same length: length
x_axis = 1:1:length; % The x-axis


plot(x_axis, (PGELLA + 80)/10, 'b-o', 'LineWidth', 1, 'MarkerSize', 8); 
hold on
plot(x_axis, (Regular_PG + 80)/10, 'r-^', 'LineWidth', 1, 'MarkerSize', 8);
box on
grid on


xlabel('Number of Time Slots', 'FontSize', 20, 'interpreter','latex')
ylabel('Average Reward of Devices', 'FontSize', 20, 'interpreter','latex')

annotation('textarrow',[0.21,0.15],[0.25,0.4],'String','Environment $1$ ', 'FontSize', 12, 'interpreter','latex')
annotation('textarrow',[0.4,0.32],[0.3,0.2],'String','Environment $2$ ', 'FontSize', 12, 'interpreter','latex')
annotation('textarrow',[0.68,0.62],[0.55,0.7],'String','Environment $3$ ', 'FontSize', 12, 'interpreter','latex')
annotation('textarrow',[0.8,0.75],[0.3,0.48],'String','Environment $4$ ', 'FontSize', 12, 'interpreter','latex')

legend('Lifelong RL','Regular PG')  %,'AbPG-ELLA')PGInterELLA
legend('Location','southeast', 'FontSize', 16, 'interpreter','latex')