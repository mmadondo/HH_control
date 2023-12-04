%% computes cost function J

function [Jc,dJ] = J(s, x, alphaG, Q, zstar)
if nargout==0
    runMinimalExample
    return
end

if not(exist('alphaG','var')) || isempty(alphaG)
    alphaG = 1.0;
end
if not(exist('Q','var')) || isempty(Q)
    Q = 1.0;
end

ds = diff(s);
nt = numel(s);


% Bz * x = z
Bz = speye(nt, nt);
Bz(end,:) = [];
Bz = kron(speye(4,5), Bz);

% Bu * x = u
Bu = speye(nt, nt);
Bu(end,:) = []; % don't penalize last time step 
Bu = kron([0 0 0 0 1], Bu);

dsbx = kron(ones(4,1),ds);
resz = Bz*x-zstar(1:end-4);
resu = Bu*x;
L = 0.5* resu'*(ds.*resu) + 0.5*Q * resz'*(dsbx.* resz);

% compute G
% It*x = z(T)
It = speye(nt,nt);
It = It(end,:); % get last component for each state variable
It = kron(speye(4,5),It);
% g = (It*x(:)  < 0.45)' * (0.45 - It*x(:))

res = It*x(:) - zstar(end-3:end);
g = alphaG*0.5*(res'*res);

Jc = L + g;

if nargout>1
    dg = alphaG*(It'*res(:));
    dL = Bu'*(ds.*resu)+ Q * Bz'*(dsbx.*resz); 
    dJ = (dg + dL)';
end


function runMinimalExample
nt = 15;

Vmstar = load('./target_solution/target_Vm.mat').Vm_out;
mstar = load('./target_solution/target_m.mat').m_out;
nstar = load('./target_solution/target_n.mat').n_out;
hstar = load('./target_solution/target_h.mat').h_out;

idx = 1:floor(size(Vmstar,2)/nt):size(Vmstar,2);
zstar = [Vmstar(:,idx)'; mstar(:,idx)'; nstar(:,idx)'; hstar(:,idx)'];  


s = linspace(0,1,nt)';
x  = [randn(nt,1); rand(nt*4, 1)]; % randn(nt*4,1);

[Jc,dJ] = feval(mfilename,s,x,[],[],zstar);
dx = randn(size(x));
dJdx = dJ*dx;
h_list = []; E0_list = []; E1_list = [];

for k=1:nt
    h = 0.5^k;
    ft = feval(mfilename, s,x+h*dx,[],[],zstar);

    E0 = norm(ft-Jc);
    E1 = norm(ft-Jc-h*dJdx);

    h_list(end+1) = h;
    E0_list(end+1) = E0;
    E1_list(end+1) = E1;
    fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',h,E0,E1)
end
loglog(h_list, E0_list, h_list, E1_list)
