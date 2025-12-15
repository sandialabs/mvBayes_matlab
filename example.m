%% generate functions
clc; clear
addpath(genpath('fdasrvf_MATLAB'))
bppr = false; % if false we default to bass
if bppr
    addpath('bayesppr_matlab/')
else
    addpath('BASS_matlab/')
end


f = @(x) normpdf(linspace(0,1,99), sin(2*pi*x(1)^2)/4 - sqrt(x(1)*x(1))/10 + .5, 0.05) * x(2);

n = 100;
nt = 99;
p = 3;
x_train = rand(n, p);
x_test = rand(1000, p);
e = randn(1, n*99);
y_train = zeros(n, 99);
for i = 1:n
    y_train(i,:) = f(x_train(i,:));
end
y_test = zeros(1000, 99);
for i = 1:1000
    y_test(i,:) = f(x_test(i,:));
end

%% generate obs
x_true = [0.1028, 0.4930];
ftilde_obs = f(x_true);
gam_obs = linspace(0, 1, nt);
vv_obs = gam_to_psi(gam_obs');

tt = linspace(0,1,nt);
out = fdawarp(y_train',tt');
out = out.multiple_align_functions(ftilde_obs',0.01);
gam_train = out.gam;
vv_train = gam_to_psi(gam_train);
ftilde_train = out.fn;
qtilde_train = out.qn;
ftilde_obs = out.fmean;

figure()
plot(tt, ftilde_obs, 'k')
hold on
plot(tt, y_train', 'Color', [0.6, 0.6, 0.6])
plot(tt, ftilde_obs, 'k', 'LineWidth',2)
legend('Experiment')

figure()
plot(tt, ftilde_obs, 'k')
hold on
plot(tt, ftilde_train, 'Color', [0.6, 0.6, 0.6])
plot(tt, ftilde_obs, 'k', 'LineWidth',2)
legend('Experiment')

figure()
plot(tt, gam_obs, 'k')
hold on
plot(tt, gam_train', 'Color', [0.6, 0.6, 0.6])
plot(tt, gam_obs, 'k', 'LineWidth',2)
axis square
legend('Experiment')

figure()
plot(tt, vv_obs, 'k')
hold on
plot(tt, vv_train', 'Color', [0.6, 0.6, 0.6])
plot(tt, vv_obs, 'k', 'LineWidth',2)
legend('Experiment')

%% Fit Emulators
if bppr
    emu_ftilde = mvBayes(@bppr, x_train, ftilde_train', 'pca', 4);
    emu_ftilde.plot()

    emu_vv = mvBayes(@bppr, x_train, vv_train, 'pns');
    emu_vv.plot()
else
    emu_ftilde = mvBayes(@bass, x_train, ftilde_train', 'pca', 4);
    emu_ftilde.plot()

    emu_vv = mvBayes(@bass, x_train, vv_train', 'pns');
    emu_vv.plot()
end
