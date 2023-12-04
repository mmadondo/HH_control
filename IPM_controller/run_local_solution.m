% 
% This is a finite-time horizon optimal control problem, written
% abstractly as
%
% min_(z,u) J(z,u) subject to z' = f(z,u), z(0) = z_0, -1<= u <= 1
% 
% Here, we discretize the ODE constraint with backward Euler and store the
% discretized values of z and u in a vector x = [z(:,1) z(:,2), u].
%
% You need the optimization toolbox and three custom files to run this
%
% J.m -- implements the objective function
% f.m -- implements the right hand side of the ODE
% c.m -- implements the time discretization of the ODE constraint

clear all; close all; clc;
nt = 1500;
T = 30;
s = linspace(0,T,nt)';

theta = [1, 380, 36, 0.3, 115, -12, 10.613];

alphaG = 1.0;
Q = 200.0;

Aeq = speye(nt);
Aeq = Aeq(1,:);
Aeq = kron(eye(4,5), Aeq);

% Z0 = linspace(-40,40,100);
% X = zeros(numel(Z0),nt,5);

% for i=1:numel(Z0)
% generate new z0 
% z0 = zeros(4,1)+[10*randn(1,1); zeros(3,1)]; % [Z0(i) 0 0 0 ]; 
z0 = zeros(4,1); % prob_xInit 
z_str = strjoin(string(z0), '_');
z = ones(nt,1)*z0';
u = zeros(nt,1);
x0 = [z(:);u];

%% generate reference data
thetaStar = [1,120,36,0.3, 115, -12, 10.613];
odefun = @(s,z) f(s,z,thetaStar);
[~,zstar] = ode45(odefun,s,z0);
% save('../experiments/local_solution/zstar.mat', 'zstar')

odefun = @(s,z) f(s,z,theta);
[~,ztraj0] = ode45(odefun,s,z0);
% x0 = [ztraj0(:);u];


%% control problem

obj = @(x) J(s,x,alphaG,Q,zstar(:));
cnl = @(x) c(s,x,theta);

options = optimoptions("fmincon",...
"Algorithm","interior-point",...
"EnableFeasibilityMode",true,...
"SubproblemAlgorithm","cg",'Display','iter',...
...
"SpecifyConstraintGradient",true,...
'SpecifyObjectiveGradient',true,'MaxFunctionEvaluations',4e5);

% lb = kron([-Inf;-Inf;-Inf;-Inf;-40], ones(nt,1));
% ub = kron([Inf;Inf;Inf;Inf;40], ones(nt,1));
lb = [];
ub = [];

x = fmincon(obj,x0, [],[],Aeq,z0,lb,ub,cnl,options);
% X(i,:,:) = reshape(x,[],5);
x = reshape(x,[],5);

% end
% save('../experiments/local_solution/sub_opt_exp/local_solution.mat','Z0','alphaG','Q','X','lb','ub','theta')


%% compare normal and pathological states
time = strrep(strrep(datestr(now), ' ', '_'), ':', '_')
save_path = '../experiments/local_solution/';

lw = 2;
figure(1); clf;
subplot(2,2,1);
plot(s,ztraj0(:,1),'-','LineWidth',lw,'DisplayName','z');
hold on
plot(s,zstar(:,1),'-','LineWidth',lw,'DisplayName','z^*');
legend()
title('Vm')
subplot(2,2,2);
plot(s,ztraj0(:,2),'-','LineWidth',lw,'DisplayName','z');
hold on
plot(s,zstar(:,2),'-','LineWidth',lw,'DisplayName','z^*');
title('m')
subplot(2,2,3);
plot(s,ztraj0(:,3),'-','LineWidth',lw,'DisplayName','z');
hold on
plot(s,zstar(:,3),'-','LineWidth',lw,'DisplayName','z^*');
title('n')
subplot(2,2,4);
plot(s,ztraj0(:,4),'-','LineWidth',lw,'DisplayName','z');
hold on
plot(s,zstar(:,4),'-','LineWidth',lw,'DisplayName','z^*');
title('h')

%%
membrane_potential = figure(2); clf;
plot(s,ztraj0(:,1),'-','LineWidth',lw,'DisplayName','z');
hold on
plot(s,zstar(:,1),'-','LineWidth',lw,'DisplayName','z^*');
xlabel('Time (ms)');
ylabel('Membrane Potential (mV)');
legend('z','z^*');
% title('Membrane Potential');
saveas(membrane_potential, strcat(save_path, sprintf('local_solution_membrane_potential_%s.png', time)));

%%
gating_variables = figure(3); clf;
plot(s,ztraj0(:,2),'-','LineWidth',lw,'DisplayName','z');
hold on;
% plot(s,zstar(:,2),'-.','LineWidth',lw,'DisplayName','z^*');
plot(s,ztraj0(:,3),'-','LineWidth',lw,'DisplayName','z');
% plot(s,zstar(:,3),'-.','LineWidth',lw,'DisplayName','z^*');
plot(s,ztraj0(:,4),'-','LineWidth',lw,'DisplayName','z');
% plot(s,zstar(:,4),'-.','LineWidth',lw,'DisplayName','z^*');
xlabel('Time (ms)');
ylabel('Activation/Inactivation');
% title('Channel Gating Variables');
legend('m', 'n', 'h')
% legend('m', 'm^*', 'n', 'n^*', 'h', 'h^*');
saveas(gating_variables, strcat(save_path, sprintf('local_solution_gating_variables_%s.png', time)));

%%
lw = 2;
figure(4); clf;
subplot(2,3,1);
plot(s,x(:,1),'-','LineWidth',lw,'DisplayName','z');
hold on
plot(s,zstar(:,1),'-','LineWidth',lw,'DisplayName','z^*');
legend()
title('Vm')
subplot(2,3,2);
plot(s,x(:,2),'-','LineWidth',lw,'DisplayName','z');
hold on
plot(s,zstar(:,2),'-','LineWidth',lw,'DisplayName','z^*');
title('m')
subplot(2,3,4);
plot(s,x(:,3),'-','LineWidth',lw,'DisplayName','z');
hold on
plot(s,zstar(:,3),'-','LineWidth',lw,'DisplayName','z^*');
title('n')
subplot(2,3,5);
plot(s,x(:,4),'-','LineWidth',lw,'DisplayName','z');
hold on
plot(s,zstar(:,4),'-','LineWidth',lw,'DisplayName','z^*');
title('h')

subplot(2,3,3)
plot(s,x(:,5),'-','LineWidth',lw,'DisplayName','z');
title('control')

%% compute ionic currents

INa = theta(2) * x(:,2).^3 .* x(:,4) .* (x(:,1) - theta(5));
IK = theta(3)*x(:,3).^4 .* (x(:,1) - theta(6));
IL = theta(4)*(x(:,1) - theta(7));

lw = 2;
ionic_currents = figure(5); clf;
plot(s, INa, 'LineWidth',lw);
hold on;
plot(s, IK, 'LineWidth',lw);
plot(s, IL, 'LineWidth',lw);
xlabel('Time (ms)');
ylabel('Current (mA/cm^2)');
% title('Ionic Currents');
legend('INa', 'IK', 'IL');
saveas(ionic_currents, strcat(save_path, sprintf('local_solution_ionic_currents_%s.png', time)));

%% save output
save(strcat(save_path, sprintf('local_solution_%s.mat', time)), 'z0', 'ztraj0', 'alphaG','Q','x','lb','ub','theta','INa','IK','IL');
