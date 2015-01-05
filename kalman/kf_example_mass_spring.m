% KFExampleMassSpring
% Setup entire simulation here first, then create function structure for
% future use.
clear all
close all

% 
m = 2;     % Measured states
L = m + 2; % States + parms
% state vector x indexed as [velocity, position, damping_coeff, spring_coeff]


% Noise definitions:
% Define Q:
q1 = 0; % Inherent unmodelable stochasticity.
qb = 0; % How much b changes over time - Vary this unknown "knob"
qc = .1; %  "  "
Q = diag([q1, 0, qb, qc]);
R = diag([.0001; .0001]); % Error in measurement

% Define parameters, and initial guess on parameters:
b     = -.25;
c     = -6;
SigP  = diag([.00000001; .00000001]);  % Uncertainty in guess of p

% Define discrete time course and measurement points
tSpan = [0 20];                     % Simulation duration
dt    = .1;                         % Interval between time points
tvec  = tSpan(1):dt:tSpan(end);     % Time at discrete time points
kvec  = 1:1:(length(tvec));         % Count of discrete time points
kend  = length(kvec);

kHasObs = zeros(1,length(kvec));    % Create boolean vector of which time points have observations. (fake data)
kHasObs(1:1:end) = 1;
% kHasObs(1:10)     = 1;

% Simulate spring and generate noisy data:
% Define model:
dynamics = @(t,x) [x(3)*x(1)+x(4)*x(2); x(1);0;0];

% Define J
J = @(X) ([X(3) X(4) X(1) X(2); 1 0 0 0; 0 0 0 0; 0 0 0 0]);

% Define H:
H = [eye(m), zeros(m)];

Ysim0 = [1; .5;b;c];
[Tsim, Ysim] = ode45(dynamics,tSpan,Ysim0);
Ysim = Ysim';
Yobs = zeros(m,length(kvec));
for i = 1:length(kvec)
    Yobs(:,i) = H * Ysim(:,find(Tsim >= tvec(i),1)) +  sqrtm(R) * randn(2,1);
end



% Initialize vectors:
Ybar         = zeros(m,kend);
Xhat         = zeros(L,kend);
Xbar         = zeros(L,kend);
SigXh        = zeros(L,L,kend);
SigXb        = zeros(L,L,kend);
SigIn        = zeros(m,m,kend);
K            = zeros(L,m,kend);

% Initialize state estimation:
k            = 1;
Xhat(:,k)    = Ysim0;
Xhat(3:4,k) = Xhat(3:4,k) + [-.4;1];
SigXh(:,:,1) = [R,zeros(2);zeros(2),SigP];

for k = kvec(2:end)
    
    % Prediction
    Ac              = J(Xhat(:,k-1));

    A               = eye(4) + dt*Ac;
    [tsim,ysim]=ode45(dynamics,[0 dt], Xhat(:,k-1));
    Xbar(:,k)=ysim(end,:)';
%     Xbar(:,k)       = A*Xhat(:,k-1); % could just directly integrate nonlinear model for dt rather than linearizing
    SigXb(:,:,k)    = A * SigXh(:,:,k-1) * A' + Q*dt;
    
    if kHasObs(k) == 1
        
        % Measurement
        Ybar(:,k)    = H * Xbar(:,k);
        SigIn(:,:,k) = H * SigXb(:,:,k) * H' + R;
        K(:,:,k)     = SigXb(:,:,k) * H' /SigIn(:,:,k);
        SigIn(:,:,k)
        Xhat(:,k)    = Xbar(:,k) + squeeze(K(:,:,k)) * (Yobs(:,k) - Ybar(:,k));
        SigXh(:,:,k) = (eye(4) - squeeze(K(:,:,k)) * H) * SigXb(:,:,k);
    else
        
        Xhat(:,k)    = Xbar(:,k);
        SigXh(:,:,k) = SigXb(:,:,k);
    end
    
end

% Plot Underlying Model, Y observation points, and Xhat estimations (and
% maybe Xbar pre-measurement predictions) over simulation
figure(2), clf, 
subplot(1,2,1), hold on, axis([0 20 -2.5 2.5])
xi=2;
plot(Tsim,Ysim(xi,:),'r')
plot(kvec(logical(kHasObs))*dt,Yobs(xi,logical(kHasObs)),'rs')
plot(kvec*dt,Xhat(xi,:),'b.')
% plot(kvec,Xbar(2,:),'c')

% Plot estimation of parameters over simulation
% figure(3), clf, 
subplot(1,2,2), hold on, axis([0 20 -10 2.5])
plot(kvec*dt,Xhat(3,:),'r')
plot(kvec*dt,Xhat(4,:),'m'), legend('b','c')