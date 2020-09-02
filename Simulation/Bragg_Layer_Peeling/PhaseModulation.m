clearvars -except 
clc
close all

%% Create the apodization profile (denoted as "qn"); note: it contains both amplitude and phase elements
dw=[10e-9]; % corrugation width
amplitude_qn=dw/5.6e-9 *1e+4; % for small dws, every 5.6-nm dw corresponds to kappa of about 1e+4 m^-1

% Import from the data created by the LPA
qn_name = 'Dispersion_less_flattop_N2486_P316';
load (qn_name);

% We can also create the apodizatin profile directly if it is simple
% Uniform
%abs_qn=amplitude_qn.*[ones(1,200)];
%pha_qn=zeros(1,length(abs_qn));

% Gaussian apodization
% zn=1:800;
% abs_qn=amplitude_qn.*exp(-4*log(2)* ( (zn - length(zn)/2 )/(290) ).^2); % Gaussian apodized
% pha_qn=zeros(1,length(abs_qn));

% pi-phase shifted
% abs_qn=amplitude_qn.*[ones(1,400)];
% pha_qn=[zeros(1,length(abs_qn)/2) pi*ones(1,length(abs_qn)/2)];



%% Basic grating parameters

w=0.5e-6; % waveguide width
period = 316e-9;
sf = 316; % sampling points per period
N=length(abs_qn);
L =period*N;



%% Validation of the complex apodization profile, by CMT
qn=abs_qn.*exp(-1j*pha_qn);
wavelength0=1546.35e-9; % wavelength center
span =16e-9; % wavelength range
nw=200; % wavelength points
wavelengths = wavelength0 + linspace(-0.5*span, 0.5*span, nw);

% CMT calculation
[detuning]=lam2det(4.195,wavelength0,wavelengths); % convert lambda to detunning
[r_re,t_re]=tmmcalc(qn,period,detuning); % CMT calculation
R_I=10*log10(abs(r_re).^2);
T_I=10*log10(abs(t_re).^2);
figure,plot(wavelengths*1e+9, R_I), title('Reflection spectrum (dB), from CMT');
figure,plot(wavelengths*1e+9, T_I),title('Transmission spectrum (dB), from CMT');
% Group delay calculation
pha_r=phase(r_re);
w_step=span/nw*1e+9; c1=w_step*1.2479e+11*2*pi/1e+12; gp=[0 diff(pha_r)/c1];
figure,plot(wavelengths*1e+9-1546, gp),title('Group-delay response (ps) in reflection, from CMT')
%return

%% Create phase modulation profile of the grating
phase_type_m={ 'REC' 'SIN' 'SAW'}; % Choose the type of the phase modulation profile

% Resample
z_re= period/sf:period/sf:L;  % "re" stands for "Resample"
N_re=length(z_re);
z_or=linspace(0,(N-1)*period,N); % "or" stands for "Original"
pha_qn_re=interp1(z_or,pha_qn,z_re,'PCHIP');
h=qn;
abs_qn_norm=abs_qn/max(abs_qn);

phase_type=phase_type_m{2};
period_pha=[2]; % period_pha: phase period (um)
period_pha=period_pha*1e-6;

% Calculate how far the side-resonances are located from the center
resonance_spacing= wavelength0/(2*4.195*period_pha/wavelength0+1);

if  strcmp( phase_type,'SIN') ==1  
    for i=1:length(abs_qn)
        f=@(x)besselj(0,1*x)-abs_qn_norm(i);
        phadep(i)=fzero(f,[0,2.40482555]);  % 'phadep' stands for the phase modulation depth;
    end
    phadep_re=interp1(z_or,phadep,z_re,'PCHIP');
    pha=phadep_re.*sin(2*pi.*z_re/period_pha) +pha_qn_re;
    
elseif    strcmp( phase_type,'REC') ==1
    for i=1:length(abs_qn)
        f=@(E) cos(E)-abs_qn_norm(i);
        phadep(i)=fzero(f,[0,pi/2]);     
    end
    phadep_re=interp1(z_or,phadep,z_re,'PCHIP');
    pha=phadep_re.*square(2*pi.*z_re/period_pha)+pha_qn_re;
       
elseif  strcmp( phase_type,'SAW') ==1
    for i=1:length(abs_qn)
        f=@(k) (sin(k)) ./ (k)-abs_qn_norm(i);
        phadep(i)=fzero(f,[eps,pi]);
    end
    phadep_re=interp1(z_or,phadep,z_re,'PCHIP');
    pha=phadep_re.*sawtooth(2*pi.*z_re/period_pha)+pha_qn_re;
    
end


%% Create vertices of the grating structure edge profiles
% use "sin" or "square" function to have different corrugation shapes
l1 = dw/2*square(z_re*2*pi/period+pha) +w/2;
l2=-l1;


%% Grating structure vadidation, using SS-TMM
dh = l1-l2;
z_um=z_re*1e+6;
l1_co=[z_um;l1*1e+6];
l2_co=fliplr([z_um;l2*1e+6]);
co=[l1_co l2_co];

sf1=30; % down-sampling based on sf; 
%Note: as we set "sf" as 316 (equal to how many nms per period), "sf1" will mean the sampling interval (nm) used in the SS-TMM
[r,t,rphase]=Fun_SS_TMM(sf,sf1,dh,span,nw,wavelength0,period);
R=10*log10(abs(r).^2);
figure,plot(wavelengths*1e+9, R_I), title('Reflection spectrum (dB), from SS-TMM');
figure,plot(wavelengths*1e+9, T_I),title('Transmission spectrum (dB), from SS-TMM');
pha_r=phase(r); gp=[0;diff(pha_r)/c1];
figure,plot(wavelengths*1e+9, gp),title('Group-delay response (ps) in reflection, from SS-TMM');


%% Create a .txt file containing the grating structure vertices 
% Create a forder
fordername=[qn_name];
mkdir(fordername);
% Write data on file
filename=[fordername '_'  phase_type   '_PP' num2str(period_pha*1e+6) '_DW' num2str(dw*1e+9)  '.txt'];
path=[cd '/' fordername];  % "cd" returns the current working path
file=[path '/' filename];
fid=fopen(file,'w');
fprintf(fid,'%3.7f, %3.7f,',co(:,(1:end-1)));
fprintf(fid,'%3.7f, %3.7f',co(:,end));
fclose(fid);













