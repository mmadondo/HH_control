% define HH model parameters
Cm = 1;     % membrane capacitance (uF/cm^2)
gNa = 120;  % maximum sodium conductance (mS/cm^2)
gK = 36.0;    % maximum potassium conductance (mS/cm^2)
gL = 0.3;   % leak conductance (mS/cm^2)

ENa = 115.0;  % sodium reversal potential (mV) 
EK = -12.0;   % potassium reversal potential (mV) 
EL = 10.613;  % leak reversal potential (mV)  

% initial conditions
Vm = 0;     % membrane potential (mV)
m = 0;      % activation variable for sodium current
h = 0;      % inactivation variable for sodium current
n = 0;      % activation variable for potassium current

% set simulation parameters
t = linspace(0,30,3000);    % simulation time vector
nt = size(t);
dt = 30/3000;      % time step (ms)
u = zeros(nt);  % external stimulus

% preallocate output vectors
Vm_out = zeros(nt);
m_out = zeros(nt);
h_out = zeros(nt);
n_out = zeros(nt);
INa_out = zeros(nt);
IK_out = zeros(nt);
IL_out = zeros(nt);

for i = 1:length(t)
    % compute ionic currents
    INa = gNa*m^3*h*(Vm - ENa);
    IK = gK*n^4*(Vm - EK);
    IL = gL*(Vm - EL);
    
    % compute derivatives of state vars
    Vm_dot = (1/Cm)*(u(i)-INa - IK - IL);
    m_dot = alpha_m(Vm)*(1 - m) - beta_m(Vm)*m;
    n_dot = alpha_n(Vm)*(1 - n) - beta_n(Vm)*n;
    h_dot = alpha_h(Vm)*(1 - h) - beta_h(Vm)*h;
    
    % update variables
    Vm = Vm + dt*Vm_dot;
    m = m + dt*m_dot;
    n = n + dt*n_dot;
    h = h + dt*h_dot;
    
    % store output
    Vm_out(i) = Vm;
    m_out(i) = m;
    h_out(i) = h;
    n_out(i) = n;
    INa_out(i) = INa;
    IK_out(i) = IK;
    IL_out(i) = IL;
end

% save solutions
target_sol = [Vm_out', m_out', n_out', h_out'];
save_path = '../target_solution';
save(strcat(save_path, 'target_sol.mat'), 'target_sol')

% plot results
subplot(2,2,1);
plot(t, Vm_out);
xlabel('Time (ms)');
ylabel('Membrane Potential (mV)');
title('Membrane Potential');

subplot(2,2,2);
plot(t, m_out);
hold on;
plot(t, n_out);
plot(t, h_out);
xlabel('Time (ms)');
ylabel('Activation/Inactivation');
title('Channel Gating Variables');
legend('m', 'n', 'h');

subplot(2,2,3);
plot(t, INa_out);
hold on;
plot(t, IK_out);
plot(t, IL_out);
xlabel('Time (ms)');
ylabel('Current (mA/cm^2)');
title('Ionic Currents');
legend('INa', 'IK', 'IL');


% helper functions for gating variable alpha and beta functions
function a = alpha_m(Vm)
    a = (2.5 - 0.1*Vm) ./ (exp(2.5 - 0.1*Vm) - 1);
end

function b = beta_m(Vm)
    b = 4*exp(-Vm/18);
end

function a = alpha_h(Vm)
    a = 0.07*exp(-Vm/20);
end

function b = beta_h(Vm)
    b = 1 ./ (exp(3 - 0.1*Vm) + 1);
end

function a = alpha_n(Vm)
    a = (0.1 - 0.01*Vm) ./ (exp(1 - 0.1*Vm) - 1);
end

function b = beta_n(Vm)
    b = 0.125*exp(-Vm/80);
end
