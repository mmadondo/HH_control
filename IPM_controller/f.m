function [fc,df] = f(s, z, theta)
    if nargout==0
        runJacobianTest
        return
    end
    
    if not(exist('theta','var')) || isempty(theta)
        theta = [1,120,36,0.3, 115, -12, 10.613];
    end

    
    % define parameters
    Cm = theta(1);     % membrane capacitance (uF/cm^2)
    gNa =theta(2);  % maximum sodium conductance (mS/cm^2)
    gK = theta(3);    % maximum potassium conductance (mS/cm^2)
    gL = theta(4);   % leak conductance (mS/cm^2)
    
    ENa = theta(5);  % sodium reversal potential (mV) 
    EK = theta(6);   % potassium reversal potential (mV) 
    EL = theta(7);  % leak reversal potential (mV) 

    z = reshape(z,[],4);
    nt = size(z, 1);
    
    % state variable
    Vm = z(:,1); % membrane potential (mV)
    m = z(:,2); % activation variable for sodium current
    n = z(:,3); % activation variable for potassium current
    h = z(:,4); % inactivation variable for sodium current
    
    % compute ionic currents
    % INa = gNa * m.^3 .* h .* (Vm - ENa);
    % IK = gK*n.^4 .* (Vm - EK);
    % IL = gL*(Vm - EL);

    % compute derivatives of state vars
    Vm_dot = (1/Cm)*(-gNa * m.^3 .* h .* (Vm - ENa) - gK*n.^4 .* (Vm - EK) - gL*(Vm - EL));  % dVm/dt
    dvdv = diag(- (1/Cm)*(gNa * m.^3 .* h + gK*n.^4 + gL));
    dvdm = diag(-(1/Cm)*(3 * gNa * m.^2 .* h .* (Vm - ENa)));
    dvdn = diag(-(1/Cm)*(4 * gK * n.^3 .* (Vm - EK)));
    dvdh = diag(-(1/Cm)*(gNa * m.^3 .* (Vm - ENa)));

    % compute gating vars and their derivatives
    [am, damdv] = alpha_m(Vm);
    [an, dandv] = alpha_n(Vm);
    [ah, dahdv] = alpha_h(Vm);
    [bm, dbmdv] = beta_m(Vm);
    [bn, dbndv] = beta_n(Vm);
    [bh, dbhdv] = beta_h(Vm);
    
    m_dot = am.*(1 - m) - bm.*m;    % dm/dt
    dmdv = diag(damdv.*(1 - m) - dbmdv.*m);
    dmdm = diag(- am - bm);
    dmdn = sparse(nt,nt);
    dmdh = sparse(nt,nt);

    n_dot = an.*(1 - n) - bn.*n;    % dn/dt
    dndv = diag(dandv.*(1 - n) - dbndv.*n);
    dndm = sparse(nt,nt);
    dndn = diag(-an - bn);
    dndh = sparse(nt,nt);

    h_dot = ah.*(1 - h) - bh.*h;    % dh/dt
    dhdv =  diag(dahdv.*(1 - h) - dbhdv.*h);
    dhdm = sparse(nt,nt);
    dhdn = sparse(nt,nt);
    dhdh = diag(-ah - bh);

    fc = [Vm_dot; m_dot; n_dot; h_dot];

    if nargout>1
        df = [dvdv dvdm dvdn dvdh;
              dmdv dmdm dmdn dmdh;
              dndv dndm dndn dndh;
              dhdv dhdm dhdn dhdh];
    end
end

function runMinimalExample
    nt = 15;
    z = [-5 + (-5+10)*rand(nt,1); rand(nt*3, 1)];
    [f0,df] = feval(mfilename,[],z);
    dz = randn(size(z));
    dfdz = df*dz;
    h_list = []; E0_list = []; E1_list = [];

    for k=1:20
        h = 0.5^k;
        ft = feval(mfilename, [],z+h*dz);
        E0 = norm(ft-f0);
        E1 = norm(ft-f0-h*dfdz);

        h_list(end+1) = h;
        E0_list(end+1) = E0;
        E1_list(end+1) = E1;
    
        fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',h,E0,E1)
    end

    loglog(h_list, E0_list, h_list, E1_list)

end

function runJacobianTest
    nt = 15;
    z = [-5 + (-5+10)*rand(nt,1); rand(nt*3, 1)];
    dz = randn(size(z));
    [E0,E1] = deal(zeros(4,4,25));
    
    I = speye(4);
    for i=1:4
        Qi = kron(I(i,:),speye(nt));
        for j=1:4
            fprintf("\n\n ----- i=%d, j=%d ------ \n\n", i, j )
            Pj = kron(I(j,:), ones(1,nt) );
            dzj = dz.*Pj(:);

            [f0,df] = feval(mfilename,[],z);
            g0 = Qi*f0;
            dgdz = Qi*(df*dzj);
            

            for k=1:size(E0,3)
                h = 0.5^k;
                ft = feval(mfilename, [],z+h*dzj);
                gt = Qi*ft;
                

                E0(i,j,k) = norm(gt-g0);
                E1(i,j,k) = norm(gt-g0-h*dgdz);

                fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',h,E0(i,j,k),E1(i,j,k))
            end
        end
    end
    idnz = max(abs(E0),[],3)>1e-16; % these are the non-zero blocks of the Jacobian
    idlin = idnz.*(max(abs(E1),[],3)<1e-10); % these are the linear blocks
    idnonlin = idnz.*(max(abs(E1),[],3)>1e-10); % these are the linear blocks
    
    cnt = sum((diff(log2(E1),1,3)<1.6) .* (abs(diff(log2(E0),1,3)-1)<0.1),3);
    incorrect_blocks = (1-cnt<3).* idnonlin
%     loglog(h_list, E0_list, h_list, E1_list)

end

% helper functions for gating variable alpha and beta functions
function [am, damdv] = alpha_m(Vm)
    num =  2.5 - 0.1*Vm;
    denom = exp(num) - 1;
    am = num ./ denom;    % alpha_m
    if nargout > 1  % deriv. of alpha_m w.r.t Vm
        damdv = -0.1 ./ denom + (0.1 * num .* exp(num)) ./ denom.^2;
    end
end

function [bm, dbmdv] = beta_m(Vm)
    bm = 4*exp(-Vm/18);
    if nargout > 1
        dbmdv = -1/18 * bm;
    end
end

function [ah, dahdv] = alpha_h(Vm)
    ah = 0.07*exp(-Vm/20);
    if nargout > 1
        dahdv = -1/20 * ah;
    end
end

function [bh, dbhdv] = beta_h(Vm)
    num = exp(3 - 0.1*Vm);
    bh = 1 ./ (num + 1);
    if nargout > 1
        dbhdv = 0.1 .* num .* bh.^2;
    end
end

function [an, dandv] = alpha_n(Vm)
    num = 0.1 - 0.01*Vm;
    denom = exp(1 - 0.1*Vm) - 1;
    an = num ./ denom;

    if nargout > 1
        dandv = (-0.01 ./ denom) + 0.1 * num .* exp(10*num) ./ denom.^2;
    end
end

function [bn, dbndv] = beta_n(Vm)
    bn = 0.125*exp(-Vm/80);
    if nargout > 1
        dbndv = -1/80 * bn;
    end
end
