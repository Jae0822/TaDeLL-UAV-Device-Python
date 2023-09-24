
figure

%--------------------------------------------------------------------------
clear
load('Case_1_All.mat')
PGELLA = Avg_Learn_AbPGmodel_Mat(4,1:20);
Regular_PG = Avg_r(4,1:20);
%Regular_PG(1,1) =  -152.9535;
PGELLA = (PGELLA + 80)/10;
Regular_PG = (Regular_PG + 80)/10;

length = size(PGELLA,2); % make sure they have the same length: length
x_axis = 1:1:length; % The x-axis:

subplot(2,2,1)

[Y1]=fastsmooth(PGELLA,6,3,1);
g1 = plot(Y1,'b--','Linewidth',2)
hold on
[Y2]=fastsmooth(Regular_PG,6,3,1);
g2 = plot(Y2,'-.r','Linewidth',2)

h1 =plot(x_axis, PGELLA, 'bo'); 
h2 = plot(x_axis, Regular_PG, 'r^');

box on
grid on

% legend('Proposed Algorithm', 'Regular Policy Gradient')  %,'AbPG-ELLA')PGInterELLA
% %legend('Proposed Algorithm', 'Regular Policy Gradient')
% legend('Location','best')

LH(1) = plot(nan, nan, 'b--o','Linewidth',1);
L{1} = 'Lifelong RL';
LH(2) = plot(nan, nan, 'r-.^','Linewidth',1);
L{2} = 'Regular PG';
legend(LH, L);
legend('Location','best', 'FontSize', 12,  'interpreter','latex')
hold off


xlabel('Number of Time Slots','FontSize', 18,  'interpreter','latex')
ylabel('Average Reward', 'FontSize', 18,  'interpreter','latex')
title('(a)')
%--------------------------------------------------------------------------
clear
load('Case_2_All.mat')
PGELLA = Avg_Learn_AbPGmodel_Mat(4,1:20);
Regular_PG = Avg_r(4,1:20);
%Regular_PG(1,1) =  -152.9535;
PGELLA = (PGELLA + 80)/10;
Regular_PG = (Regular_PG + 80)/10;

length = size(PGELLA,2); % make sure they have the same length: length
x_axis = 1:1:length; % The x-axis:

subplot(2,2,2)

[Y1]=fastsmooth(PGELLA,6,3,1);
plot(Y1,'b--','Linewidth',2)
hold on
[Y2]=fastsmooth(Regular_PG,6,3,1);
plot(Y2,'-.r','Linewidth',2)

plot(x_axis, PGELLA, 'bo'); 
plot(x_axis, Regular_PG, 'r^');

box on
grid on

% legend('Proposed Algorithm','Regular Policy Gradient')  %,'AbPG-ELLA')PGInterELLA
% legend('Location','best')

LH(1) = plot(nan, nan, 'b--o','Linewidth',1);
L{1} = 'Lifelong RL';
LH(2) = plot(nan, nan, 'r-.^','Linewidth',1);
L{2} = 'Regular PG';
legend(LH, L);
legend('Location','best', 'FontSize', 12,  'interpreter','latex')
hold off

xlabel('Number of Time Slots','FontSize', 18,  'interpreter','latex')
ylabel('Average Reward','FontSize', 18,  'interpreter','latex')
title('(b)')
%--------------------------------------------------------------------------
clear
load('Case_3_All.mat')
PGELLA = Avg_Learn_AbPGmodel_Mat(4,1:20);
Regular_PG = Avg_r(4,1:20);
%Regular_PG(1,1) =  -152.9535;
PGELLA = (PGELLA + 80)/10;
Regular_PG = (Regular_PG + 80)/10;

length = size(PGELLA,2); % make sure they have the same length: length
x_axis = 1:1:length; % The x-axis:

subplot(2,2,3)

[Y1]=fastsmooth(PGELLA,6,3,1);
plot(Y1,'b--','Linewidth',2)
hold on
[Y2]=fastsmooth(Regular_PG,6,3,1);
plot(Y2,'-.r','Linewidth',2)

plot(x_axis, PGELLA, 'bo'); 
plot(x_axis, Regular_PG, 'r^');

box on
grid on

% legend('Proposed Algorithm','Regular Policy Gradient')  %,'AbPG-ELLA')PGInterELLA
% legend('Location','best')

LH(1) = plot(nan, nan, 'b--o','Linewidth',1);
L{1} = 'Lifelong RL';
LH(2) = plot(nan, nan, 'r-.^','Linewidth',1);
L{2} = 'Regular PG';
legend(LH, L);
legend('Location','best', 'FontSize', 12,  'interpreter','latex')
hold off

xlabel('Number of Time Slots','FontSize', 18,  'interpreter','latex')
ylabel('Average Reward','FontSize', 18,  'interpreter','latex')
title('Subplot 1: sin(x)')
title('(c)')
%--------------------------------------------------------------------------
clear
load('Case_4_All.mat')
PGELLA = Avg_Learn_AbPGmodel_Mat(4,1:30);
Regular_PG = Avg_r(4,1:30);
%Regular_PG(1,1) =  -152.9535;
PGELLA = (PGELLA + 80)/10;
Regular_PG = (Regular_PG + 80)/10;


length = size(PGELLA,2); % make sure they have the same length: length
x_axis = 1:1:length; % The x-axis:

subplot(2,2,4)

[Y1]=fastsmooth(PGELLA,6,3,1);
plot(Y1,'b--','Linewidth',2)
hold on
[Y2]=fastsmooth(Regular_PG,6,3,1);
plot(Y2,'-.r','Linewidth',2)

plot(x_axis, PGELLA, 'bo'); 
plot(x_axis, Regular_PG, 'r^');

box on
grid on

% legend('Proposed Algorithm','Regular Policy Gradient')  %,'AbPG-ELLA')PGInterELLA
% legend('Location','best')

LH(1) = plot(nan, nan, 'b--o','Linewidth',1);
L{1} = 'Lifelong RL';
LH(2) = plot(nan, nan, 'r-.^','Linewidth',1);
L{2} = 'Regular PG';
legend(LH, L);
legend('Location','best', 'FontSize', 12,  'interpreter','latex')
hold off

xlabel('Number of Time Slots','FontSize', 18,  'interpreter','latex')
ylabel('Average Reward','FontSize', 18, 'interpreter','latex')
title('(d)')