% Rui Cheng,  University of British Columbia (rcheng@ece.ubc.ca)

% Design a 3-channel photonic Hilbert transformer (HT) on a 220*500 nm Si waveguide grating [R. Cheng and L. Chrostowski.  Optics Letters 43.5 (2018): 1031-1034]
% Using the layer peeling algorithim (LPA) for the grating design synthesis... [J. Skaar, et al. IEEE Journal of Quantum Electronics 37.2 (2001): 165-173]
% Note: the code will need 'phase()' function from  System Identification Toolbox
 
close all;clear all;clc   

%% define LPA parameters; 
N=3001; % total wavelength/detuning points
channel_number=3;
R=0.5; % maximum reflectivity 
ng=4.19; % group index (for 220*500 nm Si waveguide)
detuning_per_nm=10951; % detuning variation per nm change (for 220*500 nm Si waveguide); unit: m^(-1)
wavelength_r=60; %  LPA operation range in wavelength; unit: nm
detunning_r=detuning_per_nm*wavelength_r;  %  LPA operation range in detuning; unit: m^(-1)
detunning_s=detunning_r/N; % detuning step

%% define multichannel HT spectral parameters; 
flat_width_lambda=8; % Flat-top width of the entire multichannel HT in wavelength; unit: nm
flat_width=flat_width_lambda*detuning_per_nm; % convert the flat-top width from wavelength to detuning; unit: m^(-1)
np=floor(flat_width/detunning_s/(channel_number*2)); % total wavelength/detuning points for each half-single HT channel

% convert detuning into wavelength
wavelength_center=1.55e-6;
detunning=floor(-(N-1)/2:(N-1)/2).*detunning_s;
wavelengths=1./(detunning/(2*pi*ng)+1/wavelength_center);
wavelengths=wavelengths(end:-1:1);

% define the complex response of the designed three-channel HT
% channel 1
c11=ones(1,np+1).*exp(-1j*pi*1.5);
c12=ones(1,np+1).*exp(-1j*pi*0.5);
% channel 2
c21=ones(1,np+1).*exp(-1j*pi*0.5);
c22=ones(1,np+1).*exp(1j*pi*0.5);
% channel 3
c31=ones(1,np+1).*exp(1j*pi*0.5);
c32=ones(1,np+1).*exp(1j*pi*1.5);

c=[c11 c12 c21 1 c22 c31 c32];

% set the range outside the multichannel HT to be zero
n_remain=N-length(c);
r=[zeros(1,n_remain/2) c  zeros(1,n_remain/2)];

r=r*R;

% plot the  ideal response designed above
figure,plot(wavelengths,phase(r)), title('Idea phase response')
figure,plot(wavelengths,abs(r)),title('Idea amplitude response')

%% transfer the ideal response to be physically realizable 
% calculate the corresponding impulse response
h=ifft(ifftshift(r));
h=ifftshift(h);
figure,plot(real(h)),title('Idea impulse response')

% truncate the idea impulse response,
truncate_length=300; %length of the truncation wndow; should be large enough to include the strongest center part of the impulse response
window=(-truncate_length/2:truncate_length/2)+((N+1)/2);
h_w=real(h(window));
figure,plot(real(h_w)),title('Truncated impulse response');

% shift the truncated impulse response to the positive time-domain   
t=(0:length(h_w)-1)*2*pi/detunning_r; 

% calculate the physically realizable response
r_p=h_w*exp(1i*t'*detunning);
r_amplitude=10*log10(abs(r_p).^2);
r_phase=(phase(r_p));
figure,plot(wavelengths,r_amplitude),title('Physically realizable amplitude response');
figure,  plot(wavelengths,r_phase),title('Physically realizable phase response');


%% implemente LPA on the physically realizable complex response to calculate the kappa and phase profiles (q)
dz=pi/(detunning_r);  % spatial distance per point 
%L=truncate_length*dz; % Length of the grating, should be equal to truncate_length
deltak=pi/dz/N;
k=floor(-(N-1)/2:(N-1)/2)*deltak;
r(1,:)=r_p;
for n=1:truncate_length
    p(n)=sum(r(n,:))/N;
    qabs=-atanh(abs(p(n)))/dz;              
    qphase=phase(conj(p(n))); 
    q(n)=qabs*exp(1i*qphase); 
    r(n+1,:)=exp(-2i*k*dz).*(r(n,:)-p(n))./(1-conj(p(n)).*r(n,:));
end

grating_position=dz*(1:truncate_length); 
figure,plot(grating_position,abs(q)),title('Kappa  along the grating'),
xlabel('Length (m)'),ylabel('Kappa (m^(-1))');
figure,plot(grating_position,angle(q)),title('Phase along the grating'),
xlabel('Length (m)'),ylabel('Phase (rad)')

%% check if the obtained grating kappa and phase profiles (q) is correct
% using it to re-construct the grating response using the standard coupled mode
% theory-based transfer matrix method (CMT-TMM)
[rr,tt]=tmmcalc(q,dz,k); % call the CMT-TMM funtion
Reflection=10*log10(abs(rr).^2);
figure,plot(wavelengths*1e+9,Reflection),
title('Reconstructed grating amplitude spectrum'),xlim([1525 1575]),
xlabel('Wavelength (nm)'),ylabel('dB')
figure,plot(wavelengths*1e+9,phase(rr)),
title('Reconstructed grating phase spectrum'),xlim([1525 1575]),
xlabel('Wavelength (nm)'),ylabel('rad');  % to obtain the staircase like behavior of the phase response, ... 
%the linear element of the current phase profile needs to be removed

