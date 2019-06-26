%%
clear all
close all
clc
%% real data
% load('sp500_new.mat');
% load('stocks_new.mat');
% 
% Price = StPrice(end-500:end,:)';
% IndPrice = Ind(end-500:end)';
% 
% Price(167,:)=[]; Price(369,:)=[]; Price(370,:)=[]; 
% % Price = Price(1:20,:);
% 
% R = Price(:,2:end)./Price(:,1:end-1) - 1; 
% rb = IndPrice(2:end)./IndPrice(1:end-1) - 1;
% 
% [N, T] = size(R);
%% synthetic data
N = 100;
T = 500;

rng(3);
R = randn(N, T);
rb = randn(1, T);%mean(R, 1) + randn(1, T);
%%
One = ones(N,1);
IT = 1/T*(R-One*rb)*(R-One*rb)';
mu = mean(R,2);
Sigma = 1/T*(R-mu)*(R-mu)';
l = zeros(N,1); %the algorithm NOW is only valid when l>=0
h = Inf*ones(N,1);%ones(N,1);

eta = 0;
gamma_mu = 1;
gamma_sigma = 1;
rng('shuffle')
rng(5)
lambda = 1/700*sort((1-0)*rand(N,1)+0,'descend');%1/N^5*(N - (0:1:N-1))'; %zeros(N,1); 
%%
cvx_begin quiet
    variable w_cvx(N,1)
    variable z_cvx(N,1)
    minimize( eta*0.5*w_cvx'*IT*w_cvx - gamma_mu*w_cvx'*mu ...
        + gamma_sigma*0.5*w_cvx'*Sigma*w_cvx )%+ lambda'*z_cvx )
    subject to
        One' * w_cvx == 1
        w_cvx >= l
%         w_cvx <= h
%         for i = 1:N
%             sum(z_cvx(1:i)) >= max(sum(nchoosek(w_cvx,i),2))
%         end
cvx_end
% obj_cvx = 0.5*w_cvx'*IT*w_cvx - gamma_mu*w_cvx'*mu + gamma_sigma*0.5*w_cvx'*Sigma*w_cvx ...
%     + lambda'*sort(w_cvx,'descend');
%%
rho = 0.5;
psi = max(eig(eta*IT+gamma_sigma*Sigma+rho*(One*One')));

MaxItr = 5e3;
eps = 1e-6;

itr = 1;
w = 1/N*ones(N,1);
z = w;
v = w;
u = 0;

norm_wz(itr) = norm(w-z);
AL(itr) = eta*0.5*w'*IT*w - gamma_mu*w'*mu + gamma_sigma*0.5*w'*Sigma*w + lambda'*sort(z,'descend')...
    + 0.5*rho*((One'*w+u-1)^2+norm(w+v-z)^2);
obj(itr) = eta*0.5*w'*IT*w - gamma_mu*w'*mu + gamma_sigma*0.5*w'*Sigma*w + lambda'*sort(z,'descend');

%%
for itr = 1:MaxItr
    w_old = w;
    z_old = z;
    u_old = u;
    v_old = v;
    AL_old = AL(itr);
    
    itr = itr + 1
%     m = -1/rho*IT*w + gamma_mu*1/rho*mu - gamma_sigma*1/rho*Sigma*w ...
%         - One*One'*w - (u-1)*One - v + z;
%     w = min(max(0,m),h);

%     cvx_begin quiet
%     variable w_temp(N,1)
%     minimize( 0.5*w_temp'*IT*w_temp - gamma_mu*w_temp'*mu ...
%         + gamma_sigma*0.5*w_temp'*Sigma*w_temp ...
%         + 0.5*rho*((One'*w_temp+u-1)^2+power(2,norm(w_temp+v-z))) )
%     subject to
%         w_temp >= l
%         w_temp <= h
%     cvx_end
%     w = w_temp;

    m = 1/(psi+rho)*( -(eta*IT+gamma_sigma*Sigma+rho*One*One'-psi*eye(N))*w + gamma_mu*mu ...
         - rho*(u-1)*One - rho*(v-z) );
    w = min(max(l,m),h);
        
    n = w + v;
%     z = proxSortedL1(n, lambda/rho);

    [n_s,idx] = sort(n,'descend');
%     z_s = proxSortedL1Mex(n_s,lambda/rho);
%     z(idx) = z_s;
    
%     cvx_begin quiet
%     variable z_temp(N,1)
%     variable zz_temp(N,1)
%     minimize( 0.5*power(norm(n-z_temp),2) + (lambda/rho)'*zz_temp )
%     subject to
%         for i = 1:N
%             sum(zz_temp(1:i)) >= max(sum(nchoosek(z_temp,i),2))
%         end
%     cvx_end
%     z = z_temp
    
%     cvx_begin quiet
%     variable z_temp(N,1)
%     minimize( 0.5*sum_square_abs(n_s-z_temp) + (lambda/rho)'*z_temp )
%     subject to
%         z_temp(1:N-1)-z_temp(2:N) >= 0
%     cvx_end
%     z(idx) = z_temp
    
    z_temp = lsqisotonic(N+1-(1:1:N),n_s-lambda/rho,One);
    z(idx) = z_temp;
    
    u = u + One'*w - 1;
    v = v + w - z;
    
    norm_wz(itr) = norm(w-z);
    AL(itr) = eta*0.5*w'*IT*w - gamma_mu*w'*mu + gamma_sigma*0.5*w'*Sigma*w + lambda'*sort(z,'descend')...
        + 0.5*rho*((One'*w+u-1)^2+norm(w+v-z)^2);
    obj(itr) = eta*0.5*w'*IT*w - gamma_mu*w'*mu + gamma_sigma*0.5*w'*Sigma*w + lambda'*sort(z,'descend');

    if norm(w_old-w)<eps && norm(z_old-z)<eps && norm(u_old-u)<eps && norm(v_old-v)<eps && norm(AL_old-AL(itr))<eps
        break
    end   
end

%%
figure
NS=sum(abs(w)>=1e-6);
bar(w)
txt = {['# of assets = ',num2str(NS)]};
text(20,max(w),txt)
%
%%
figure
NS=sum(abs(w_cvx)>=1e-6);
bar(w_cvx)
txt = {['# of assets = ',num2str(NS)]};
text(20,max(w_cvx),txt)