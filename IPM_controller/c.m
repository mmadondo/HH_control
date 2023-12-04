function [cineq,ceq,dcineq,dceq] = c(s,x,theta)
if nargout==0
    runMinimalExample
    return
end
cineq=[];

if not(exist('theta','var'))
    theta = [];
end

x = reshape(x,[],5);
z = x(:,1:4);
ds = diff(s);
nt = numel(s);

% Dt*x = dz/dt
dt = spdiags((1./ds)*[-1,1], 0:1, nt-1,nt);
Dt = kron(speye(4,5),dt);

% It*f = f(s,z)
It = speye(nt,nt);
It(1,:) = []; % implicit time step
It = kron(speye(4), It);

% B * x = u
Bu = speye(nt,nt);
Bu(end,:) = []; 
Bu = kron([zeros(4,4) speye(4,1)],Bu);

[fc,df] = f(s,z,theta);
ceq = Dt*x(:) - (It*fc + Bu*x(:) ); %  B*x(:)

if nargout>1
    dceq = (Dt - [It*df sparse(4*nt-4,nt)] - Bu)';
    dcineq=[];
end


function runMinimalExample
nt = 10;
s = linspace(0,1,nt)';
x  = randn(nt*5,1);
[~,f0,~,df] = c(s,x);
dx = randn(size(x));
dfdx = df'*dx;
h_list = []; E0_list = []; E1_list = [];

for k=1:20
    h = 0.5^k;
    [~,ft] = feval(mfilename, s,x+h*dx);

    E0 = norm(ft-f0);
    E1 = norm(ft-f0-h*dfdx);
    h_list(end+1) = h;
    E0_list(end+1) = E0;
    E1_list(end+1) = E1;
    fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',h,E0,E1)
end

loglog(h_list, E0_list, h_list, E1_list)
